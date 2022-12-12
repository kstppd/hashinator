/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
#pragma once
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "definitions.h"
#include "../splitvector/splitvec.h"
#include "../splitvector/split_tools.h"
#include "hash_pair.h"


template <typename T, typename U,T EMPTYBUCKET = vmesh::INVALID_GLOBALID,T TOMBSTONE = EMPTYBUCKET - 1>
struct Tombstone_Predicate{
   __host__ __device__
   inline bool operator()( hash_pair<T,U>& element)const{
      if (element.first==TOMBSTONE){element.first=EMPTYBUCKET;return false;}
      return element.offset>0;
   }
};

template<typename GID>
__device__
static inline uint32_t ext_fibonacci_hash(GID in,const int sizePower){
   in ^= in >> (32 - sizePower);
   uint32_t retval = (uint64_t)(in * 2654435769ul) >> (32 - sizePower);
   return retval;
}

template<typename GID, typename LID,GID EMPTYBUCKET=vmesh::INVALID_GLOBALID,size_t BLOCKSIZE=1024, size_t WARP=32>
__global__ 
void hasher(hash_pair<GID, LID>* src, hash_pair<GID, LID>* dst, const int sizePower,int maxoverflow){

   __shared__ int warp_exit_flag;
   const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
   const unsigned int wid = tid/WARP;
   const unsigned int w_tid=tid%WARP;

   hash_pair<GID,LID>&candidate=src[wid];
   int bitMask = (1 <<(sizePower )) - 1; 
   uint32_t hashIndex = ext_fibonacci_hash(candidate.first,sizePower);
   uint32_t optimalindex=(hashIndex) & bitMask;
   //Reset the element to EMPTYBUCKET
   if (w_tid==0){
      dst[optimalindex+candidate.offset].first=EMPTYBUCKET;
      warp_exit_flag=0;
   }

   for (size_t warp_offset=0; warp_offset<(1<<sizePower); warp_offset+=WARP){
      uint32_t vecindex=(hashIndex+ warp_offset+ w_tid) & bitMask;
      //vote for available emptybuckets in warp region
      uint32_t  mask= __ballot_sync(SPLIT_VOTING_MASK,dst[vecindex].first==EMPTYBUCKET);
      while(mask!=0 && warp_exit_flag==0){
         //LSB is the winner (little endian)--> smallest overflow candidate thread
         int winner =__ffs ( mask ) -1;
         if (w_tid==winner){
            GID old = atomicCAS(&dst[vecindex].first, EMPTYBUCKET, candidate.first);
            if (old == EMPTYBUCKET){
               atomicExch(&dst[vecindex].second,candidate.second);
               //no need to commmunicate new overflow as it will be less than before...
               int overflow = vecindex-optimalindex;
               atomicExch(&dst[vecindex].offset,overflow);
               assert(overflow<=maxoverflow && "Thread exceeded max overflow. This does fail after all!");
               warp_exit_flag=1;
            }else{
               mask &= ~(1<< winner);
            }
         }
         __syncthreads();
         if(warp_exit_flag==1){
            return;
         }
      }
   }
}




// Open bucket power-of-two sized hash table with multiplicative fibonacci hashing
template <typename GID, typename LID, int maxBucketOverflow = 32, GID EMPTYBUCKET = vmesh::INVALID_GLOBALID,GID TOMBSTONE = EMPTYBUCKET - 1 > 
class Hashinator {
private:
   //CUDA device handles
   int* d_sizePower;
   int* d_maxBucketOverflow;
   int postDevice_maxBucketOverflow;
   size_t* d_fill;
   size_t* d_tombstoneCounter;
   size_t tombstoneCounter;
   Hashinator* device_map;
   //~CUDA device handles

   //Host members
   int sizePower; // Logarithm (base two) of the size of the table
   int cpu_maxBucketOverflow;
   size_t fill;   // Number of filled buckets
   //~Host members
   
   // Fibonacci hash function for 64bit values
   __host__ __device__
   uint32_t fibonacci_hash(GID in) const {
      in ^= in >> (32 - sizePower);
      uint32_t retval = (uint64_t)(in * 2654435769ul) >> (32 - sizePower);
      return retval;
   }

    //Hash a chunk of memory using fnv_1a
   __host__ __device__
   uint32_t fnv_1a(const void* chunk, size_t bytes)const{
       assert(chunk);
       uint32_t h = 2166136261ul;
       const unsigned char* ptr = (const unsigned char*)chunk;
       while (bytes--){
          h = (h ^ *ptr++) * 16777619ul;
       }
       return h ;
    }

    // Wrapper over available hash functions 
   __host__ __device__
   uint32_t hash(GID in) const {
       static constexpr bool n = (std::is_arithmetic<GID>::value && sizeof(GID) <= sizeof(uint32_t));
       if (n) {
          return fibonacci_hash(in);
       } else {
          return fnv_1a(&in, sizeof(GID));
       }
    }
   
   // Used by the constructors. Preallocates the device pointer and bookeepping info for later use on device. 
   // This helps in reducing the number of calls to cudaMalloc
   __host__
   void preallocate_device_handles(){
      cudaMalloc((void **)&d_sizePower, sizeof(int));
      cudaMalloc((void **)&d_maxBucketOverflow, sizeof(int));
      cudaMalloc((void **)&d_fill, sizeof(size_t));
      cudaMalloc((void **)&d_tombstoneCounter, sizeof(size_t));
      cudaMalloc((void **)&device_map, sizeof(Hashinator));
   }

   // Deallocates the bookeepping info and the device pointer
   __host__
   void deallocate_device_handles(){
      cudaFree(device_map);
      cudaFree(d_sizePower);
      cudaFree(d_maxBucketOverflow);
      cudaFree(d_fill);
      cudaFree(d_tombstoneCounter);
   }

   __host__
   int kval_overflow(GID key,size_t index){
      int bitMask = (1 << sizePower) - 1;
      uint32_t hashIndex = hash(key)&bitMask;
      uint32_t optimalIndex=((index)&bitMask);
      return optimalIndex-hashIndex;
   }


   __host__
   void clean_by_compaction(){

      // TODO tune parameters based on output size
      split::SplitVector<hash_pair<GID, LID>> overflownElements(1 << sizePower, {EMPTYBUCKET, LID()});
      split_tools::copy_if<hash_pair<GID, LID>,Tombstone_Predicate<GID,LID>,32,32>(buckets,overflownElements,Tombstone_Predicate<GID,LID>());
      hasher<GID,LID><<<1,32>>> (overflownElements.data(),buckets.data(),sizePower,maxBucketOverflow);
      cudaDeviceSynchronize();
   }

   __host__ 
   void clean_tombstones_in_place(){
      while(tombstoneCounter){
         size_t i=0;
         size_t start=0;
         size_t end=0;
         if (buckets.back().first==TOMBSTONE){
            buckets.back().first=EMPTYBUCKET;
            tombstoneCounter--;
         }
         while(i<buckets.size()-1){
            const auto& current=buckets[i];
            size_t offset=1;
            if (current.first != TOMBSTONE && kval_overflow(current.first,i)==0){
               i++;
               continue;
            }
            auto next=buckets[i+offset];
            while(next.first!=EMPTYBUCKET){
               //Is this element in the correct location?
               auto  distance=kval_overflow(next.first,i+offset);
               if (distance==0) {
                  break;
               }
               offset++;
               if (i+offset>buckets.size()){
                  break;
               }
               next=buckets[i+offset];
            }
            start=i;
            end=start+offset;
            //printf("Cleaning subdomain %d to %d\n",start,end);
            //dump_buckets();
            for( size_t sub_index=start; sub_index<=end; ++sub_index){
               auto& candidate=buckets[sub_index];
               if (candidate.first==TOMBSTONE){
                  candidate.first=EMPTYBUCKET;
                  tombstoneCounter--;
                  continue;
               }
               size_t running_index= sub_index;
               auto distance=kval_overflow(candidate.first,running_index);
               size_t target_index=running_index-distance;
               while(distance){
                  auto& targetBucket=buckets.at(target_index);
                  if (targetBucket.first==EMPTYBUCKET){
                     targetBucket.first=candidate.first;
                     targetBucket.second=kval_overflow(targetBucket.first,target_index)+1;
                     candidate.first=EMPTYBUCKET;
                     break;
                  }
                  target_index++;
               }

            }
            //dump_buckets();
            i=end+1;
         }
      }
   }

   __host__
   inline bool isEmpty(const hash_pair<GID,LID>& b ){
      return b.first==EMPTYBUCKET;
   }

   __host__
   inline bool isTombStone(const hash_pair<GID,LID>& b ){
      return b.first==TOMBSTONE;
   }
   

   //Simply remove all tombstones on host by rehashing
   __host__
   void clean_tombstones(){
      //clean_by_compaction();
      clean_tombstones_in_place();
      assert(tombstone_count()==0 && "Tombstones leaked into CPU hashmap!");
   }


public:
   split::SplitVector<hash_pair<GID, LID>> buckets;
    
   __host__
   Hashinator()
       : sizePower(4), fill(0), buckets(1 << sizePower, hash_pair<GID, LID>(EMPTYBUCKET, LID())){
         preallocate_device_handles();
       };

   __host__
   Hashinator(int sizepower)
       : sizePower(sizepower), fill(0), buckets(1 << sizepower, hash_pair<GID, LID>(EMPTYBUCKET, LID())){
         preallocate_device_handles();
       };
   __host__
   Hashinator(const Hashinator<GID, LID>& other)
       : sizePower(other.sizePower), fill(other.fill), buckets(other.buckets){
         preallocate_device_handles();
       };
   __host__
   ~Hashinator(){     
      deallocate_device_handles();
   };



   // Resize the table to fit more things. This is automatically invoked once
   // maxBucketOverflow has triggered. This can only be done on host (so far)
   __host__
   void rehash(int newSizePower) {
#ifdef HASHMAPDEBUG
      std::cout<<"Rehashing to "<<( 1<<newSizePower )<<std::endl;
#endif
      if (newSizePower > 32) {
         throw std::out_of_range("Hashinator ran into rehashing catastrophe and exceeded 32bit buckets.");
      }
      split::SplitVector<hash_pair<GID, LID>> newBuckets(1 << newSizePower,
                                                  hash_pair<GID, LID>(EMPTYBUCKET, LID()));
      sizePower = newSizePower;
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size

      // Iterate through all old elements and rehash them into the new array.
      for (auto& e : buckets) {
         // Skip empty buckets ; We also check for TOMBSTONE elements
         // as we might be coming off a kernel that overflew the hashmap
         if (e.first == EMPTYBUCKET || e.first==TOMBSTONE) {
            continue;
         }

         uint32_t newHash = hash(e.first);
         bool found = false;
         for (int i = 0; i < maxBucketOverflow; i++) {
            hash_pair<GID, LID>& candidate = newBuckets[(newHash + i) & bitMask];
            if (candidate.first == EMPTYBUCKET) {
               // Found an empty bucket, assign that one.
               e.offset=i;
               candidate = e;
               found = true;
               break;
            }
         }

         if (!found) {
            // Having arrived here means that we unsuccessfully rehashed and
            // are *still* overflowing our buckets. So we need to try again with a bigger one.
            return rehash(newSizePower + 1);
         }
      }

      // Replace our buckets with the new ones
      buckets = newBuckets;
   }

   // Element access (by reference). Nonexistent elements get created.
   __host__
   LID& _at(const GID& key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate.first == key) {
            // Found a match, return that
            return candidate.second;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, assign and return that.
            candidate.first = key;
            fill++;
            candidate.offset=i;
            return candidate.second;
         }
      }

      // Not found, and we have no free slots to create a new one. So we need to rehash to a larger size.
      rehash(sizePower + 1);
      return at(key); // Recursive tail call to try again with larger table.
   }

   __host__
   const LID& _at(const GID& key) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate.first == key) {
            // Found a match, return that
            return candidate.second;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, so error.
            throw std::out_of_range("Element not found in Hashinator.at");
         }
      }

      // Not found, so error.
      throw std::out_of_range("Element not found in Hashinator.at");
   }


   //---------------------------------------

   // For STL compatibility: size(), bucket_count(), count(GID), clear()
   __host__
   size_t size() const { return fill; }

   __host__ __device__
   size_t bucket_count() const {
      return buckets.size();
   }
   
   __host__
   float load_factor() const {return (float)size()/bucket_count();}

   __host__
   size_t count(const GID& key) const {
      if (find(key) != end()) {
         return 1;
      } else {
         return 0;
      }
   }

   __host__
   void clear() {
      buckets = split::SplitVector<hash_pair<GID, LID>>(1 << sizePower, {EMPTYBUCKET, LID()});
      fill = 0;
   }

   //Try to grow our buckets until we avhieve a targetLF load factor
   __host__
   void resize_to_lf(float targetLF=0.5){
      while (load_factor() > targetLF){
         rehash(sizePower+1);
      }
   }

   __host__
   void resize(int newSizePower){
      rehash(newSizePower);     
   }


   /**
    * Host that return a device pointer that can be passed to CUDA kernels
    * The pointer is internally cleaned up by the destructors.
    * User must call clean_up_after_device() after exiting the kernel 
    * otherwise the hasmap might end up in an ill state.
    */
   __host__
   Hashinator* upload(cudaStream_t stream = 0 ){
      cpu_maxBucketOverflow=maxBucketOverflow;
      this->buckets.optimizeGPU(stream); //already async so can be overlapped if used with streams
      cudaMemcpyAsync(d_sizePower, &sizePower, sizeof(int),cudaMemcpyHostToDevice,stream);
      cudaMemcpyAsync(d_maxBucketOverflow,&cpu_maxBucketOverflow, sizeof(int),cudaMemcpyHostToDevice,stream);
      cudaMemcpyAsync(d_fill, &fill, sizeof(size_t),cudaMemcpyHostToDevice,stream);
      cudaMemcpyAsync(device_map, this, sizeof(Hashinator),cudaMemcpyHostToDevice,stream);
      cudaMemsetAsync(d_tombstoneCounter, 0, sizeof(size_t));
      return device_map;
   }

   //Just return the device pointer. Upload should be called fist 
   //othewise map bookeepping info will not be updated on device.
   __host__
   Hashinator* get_device_pointer(){
      return device_map;
   }


   /**
    * This must be called after exiting a CUDA kernel. These functions
    * will do the following :
    *  • handle communicating bookeepping info back to host. 
    *  • If the hashmap has overflown on device it will try 
    *     to get the overflow limits down to the default. 
    * */ 
   __host__
   void download(cudaStream_t stream = 0){
      //Copy over fill as it might have changed
      cudaMemcpyAsync(&fill, d_fill, sizeof(size_t),cudaMemcpyDeviceToHost,stream);
      cudaMemcpyAsync(&tombstoneCounter, d_tombstoneCounter, sizeof(size_t),cudaMemcpyDeviceToHost,stream);
      cudaMemcpyAsync(&postDevice_maxBucketOverflow, d_maxBucketOverflow, sizeof(int),cudaMemcpyDeviceToHost,stream);
#ifdef HASHMAPDEBUG
      //dump_buckets();
      std::cout<<"Cleaning TombStones"<<std::endl;
      std::cout<<"Overflow Limits Dev/Host "<<maxBucketOverflow<<"--> "<<postDevice_maxBucketOverflow<<std::endl;
      std::cout<<"Fill after device = "<<fill<<std::endl;
#endif
      this->buckets.optimizeCPU(stream);
      if (postDevice_maxBucketOverflow>maxBucketOverflow){
         rehash(sizePower+1);
      }else{
         std::cout<<"Before : "<<tombstone_count()<<" tombstones\n";
         if(tombstone_count()>0){
            clean_tombstones();
         }
         std::cout<<"After : "<<tombstone_count()<<" tombstones\n";
      }
   }

   #ifdef HASHMAPDEBUG
   __host__
   void print_all(){
      std::cout<<">>>>*********************************"<<std::endl;
      std::cout<<"Map contains "<<bucket_count()<<" buckets"<<std::endl;
      std::cout<<"Map fill is "<<fill<<std::endl;
      std::cout<<"Map size is "<<size()<<std::endl;
      std::cout<<"Map LF is "<<load_factor()<<std::endl;
      std::cout<<"Map Overflow Limit after DEVICE is "<<postDevice_maxBucketOverflow<<std::endl;
      std::cout<<"<<<<*********************************"<<std::endl;
   }

   __host__
      void print_pair(const hash_pair<GID, LID>& i){
         if (i.first==TOMBSTONE){
            std::cout<<"[╀,-,-] ";
         }else if (i.first == EMPTYBUCKET){
            std::cout<<"[▢,-,-] ";
         }
         else{
            printf("[%d,%d-%d] ",i.first,i.second,i.offset);
         }
      }
   __host__
   void dump_buckets(){
      std::cout<<"\n";
      for  (auto i :buckets){
         print_pair(i);
      }
      std::cout<<std::endl;

   }
   __host__
   void print_kvals(){
      dump_buckets();
      std::cout<<"Total Tombstones= "<<tombstone_count()<<std::endl;
   }
   __host__
   size_t tombstone_count(){
      return tombstoneCounter;
   }
   #endif

   __host__
   void swap(Hashinator<GID, LID>& other) {
      buckets.swap(other.buckets);
      int tempSizePower = sizePower;
      sizePower = other.sizePower;
      other.sizePower = tempSizePower;
      size_t tempFill = fill;
      fill = other.fill;
      other.fill = tempFill;
      std::swap(d_sizePower,other.d_sizePower);
      std::swap(d_maxBucketOverflow,other.d_maxBucketOverflow);
      std::swap(d_fill,other.d_fill);
      std::swap(device_map,other.device_map);
   }



   //Read only  access to reference. 
   __host__
   const LID& at(const GID& key) const {
      return _at(key);
   }

   //See _at(key)
   __host__
   LID& at(const GID& key) {
      return _at(key);
   }

   // Typical array-like access with [] operator
   __host__
   LID& operator[](const GID& key) {
      return at(key); 
   }


   __device__
   void set_element(const GID& key,LID val){
      size_t thread_overflowLookup;
      insert_element(key,val,thread_overflowLookup);
      atomicMax(d_maxBucketOverflow,thread_overflowLookup);
   }

   __device__
   const LID& read_element(const GID& key) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < *d_maxBucketOverflow; i++) {
         uint32_t vecindex=(hashIndex + i) & bitMask;
         const hash_pair<GID, LID>& candidate = buckets[vecindex];
         if (candidate.first == key) {
            // Found a match, return that
            return candidate.second;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, so error.
             assert(false && "Key does not exist");
         }
      }
       assert(false && "Key does not exist");
       return ; //to get the compiler not to yell
   }
   




//Host/Device Overloads
/**
                                         _   _  ___  ____ _____    ____ ___  ____  _____              
                           __/\____/\__ | | | |/ _ \/ ___|_   _|  / ___/ _ \|  _ \| ____| __/\____/\__
                           \    /\    / | |_| | | | \___ \ | |   | |  | | | | | | |  _|   \    /\    /
                           /_  _\/_  _\ |  _  | |_| |___) || |   | |__| |_| | |_| | |___  /_  _\/_  _\
                             \/    \/   |_| |_|\___/|____/ |_|    \____\___/|____/|_____|   \/    \/  
                                                 
*/
#ifndef __CUDA_ARCH__
   
   // Iterator type. Iterates through all non-empty buckets.
   __host__
   class iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<GID, LID>> {
      Hashinator<GID, LID>* hashtable;
      size_t index;

   public:
      __host__
      iterator(Hashinator<GID, LID>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

      __host__
      iterator& operator++() {
         index++;
         while(index < hashtable->buckets.size()){
            if (hashtable->buckets[index].first != EMPTYBUCKET){
               break;
            }
            index++;
         }
         return *this;
      }
      
      __host__
      iterator operator++(int) { // Postfix version
         iterator temp = *this;
         ++(*this);
         return temp;
      }
      __host__
      bool operator==(iterator other) const {
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      __host__
      bool operator!=(iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      __host__
      hash_pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      __host__
      hash_pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }
      __host__
      size_t getIndex() { return index; }
   };

   // Const iterator.
   __host__
   class const_iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<GID, LID>> {
      const Hashinator<GID, LID>* hashtable;
      size_t index;

   public:
      __host__
      explicit const_iterator(const Hashinator<GID, LID>& hashtable, size_t index)
          : hashtable(&hashtable), index(index) {}
      __host__
      const_iterator& operator++() {
         index++;
         while(index < hashtable->buckets.size()){
            if (hashtable->buckets[index].first != EMPTYBUCKET){
               break;
            }
            index++;
         }
         return *this;
      }
      __host__
      const_iterator operator++(int) { // Postfix version
         const_iterator temp = *this;
         ++(*this);
         return temp;
      }
      __host__
      bool operator==(const_iterator other) const {
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      __host__
      bool operator!=(const_iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      __host__
      const hash_pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      __host__
      const hash_pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }
      __host__
      size_t getIndex() { return index; }
   };

   // Element access by iterator
   __host__
   const const_iterator find(GID key) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate.first == key) {
            // Found a match, return that
            return const_iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return end();
         }
      }

      // Not found
      return end();
   }

   __host__
   iterator find(GID key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate.first == key) {
            // Found a match, return that
            return iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return end();
         }
      }

      // Not found
      return end();
   }
   
   __host__
   iterator begin() {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i].first != EMPTYBUCKET) {
            return iterator(*this, i);
         }
      }
      return end();
   }

   __host__
   const_iterator begin() const {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i].first != EMPTYBUCKET) {
            return const_iterator(*this, i);
         }
      }
      return end();
   }

   __host__
   iterator end() { return iterator(*this, buckets.size()); }

   __host__
   const_iterator end() const { return const_iterator(*this, buckets.size()); }

   // Remove one element from the hash table.
   __host__
   iterator erase(iterator keyPos) {
      // Due to overflowing buckets, this might require moving quite a bit of stuff around.
      size_t index = keyPos.getIndex();

      if (buckets[index].first != EMPTYBUCKET) {
         // Decrease fill count
         fill--;

         // Clear the element itself.
         buckets[index].first = EMPTYBUCKET;

         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         size_t targetPos = index;
         // Search ahead to verify items are in correct places (until empty bucket is found)
         for (unsigned int i = 1; i < fill; i++) {
            GID nextBucket = buckets[(index + i)&bitMask].first;
            if (nextBucket == EMPTYBUCKET) {
               // The next bucket is empty, we are done.
               break;
            }
            // Found an entry: is it in the correct bucket?
            uint32_t hashIndex = hash(nextBucket);
            if ((hashIndex&bitMask) != ((index + i)&bitMask)) {
               // This entry has overflown. Now check if it should be moved:
               uint32_t distance =  ((targetPos - hashIndex + (1<<sizePower) )&bitMask);
               if (distance < maxBucketOverflow) {
                  // Copy this entry to the current newly empty bucket, then continue with deleting
                  // this overflown entry and continue searching for overflown entries
                  LID moveValue = buckets[(index+i)&bitMask].second;
                  buckets[targetPos] = hash_pair<GID, LID>(nextBucket,moveValue);
                  targetPos = ((index+i)&bitMask);
                  buckets[targetPos].first = EMPTYBUCKET;
               }
            }
         }
      }
      // return the next valid bucket member
      ++keyPos;
      return keyPos;
   }

   __host__
   hash_pair<iterator, bool> insert(hash_pair<GID, LID> newEntry) {
      bool found = find(newEntry.first) != end();
      if (!found) {
         at(newEntry.first) = newEntry.second;
      }
      return hash_pair<iterator, bool>(find(newEntry.first), !found);
   }

   __host__
   size_t erase(const GID& key) {
      iterator element = find(key);
      if(element == end()) {
         return 0;
      } else {
         erase(element);
         return 1;
      }
   }
   
#else

/**
                                         ____  _______     _____ ____ _____    ____ ___  ____  _____ 
                           __/\____/\__ |  _ \| ____\ \   / /_ _/ ___| ____|  / ___/ _ \|  _ \| ____|  __/\____/\__
                           \    /\    / | | | |  _|  \ \ / / | | |   |  _|   | |  | | | | | | |  _|    \    /\    /
                           /_  _\/_  _\ | |_| | |___  \ V /  | | |___| |___  | |__| |_| | |_| | |___   /_  _\/_  _\
                             \/    \/   |____/|_____|  \_/  |___\____|_____|  \____\___/|____/|_____|    \/    \/  
*/


   // Device Iterator type. Iterates through all non-empty buckets.
   __device__
   class iterator  {
   private:
      size_t index;
      Hashinator<GID, LID>* hashtable;
   public:
      __device__
      iterator(Hashinator<GID, LID>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
      
      __device__
      size_t getIndex() { return index; }
     
      __device__
      iterator& operator++() {
         index++;
         while(index < hashtable->buckets.size()){
            if (hashtable->buckets[index].first != EMPTYBUCKET&&
                hashtable->buckets[index].first != TOMBSTONE){
               break;
            }
            index++;
         }
         return *this;
      }
      
      __device__
      iterator operator++(int){
         iterator temp = *this;
         ++(*this);
         return temp;
      }
      
      __device__
      bool operator==(iterator other) const {
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      __device__
      bool operator!=(iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      
      __device__
      hash_pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      __device__
      hash_pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }

   };


   __device__
   class const_iterator  {
   private:
      size_t index;
      const Hashinator<GID, LID>* hashtable;
   public:
      __device__
      explicit const_iterator(const Hashinator<GID, LID>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
      
      __device__
      size_t getIndex() { return index; }
     
      __device__
      const_iterator& operator++() {
         index++;
         while(index < hashtable->buckets.size()){
            if (hashtable->buckets[index].first != EMPTYBUCKET &&
                hashtable->buckets[index].first != TOMBSTONE ){
               break;
            }
            index++;
         }
         return *this;
      }
      
      __device__
      const_iterator operator++(int){
         const_iterator temp = *this;
         ++(*this);
         return temp;
      }
      
      __device__
      bool operator==(const_iterator other) const {
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      __device__
      bool operator!=(const_iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      
      __device__
      const hash_pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      __device__
      const hash_pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }
   };


   // Element access by iterator
   __device__ 
   iterator find(GID key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < *d_maxBucketOverflow; i++) {
         const hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];

         if (candidate.first==TOMBSTONE){continue;}

         if (candidate.first == key) {
            // Found a match, return that
            return iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return end();
         }
      }

      // Not found
      return end();
   }

   __device__ 
   const const_iterator find(GID key)const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < *d_maxBucketOverflow; i++) {
         const hash_pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];

         if (candidate.first==TOMBSTONE){continue;}

         if (candidate.first == key) {
            // Found a match, return that
            return const_iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return end();
         }
      }

      // Not found
      return end();
   }


   __device__
   iterator end() { return iterator(*this, buckets.size()); }

   __device__
   const_iterator end()const  { return const_iterator(*this, buckets.size()); }

   __device__
   iterator begin() {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i].first != EMPTYBUCKET) {
            return iterator(*this, i);
         }
      }
      return end();
   }


   __device__
   size_t erase(const GID& key) {
      iterator element = find(key);
      if(element == end()) {
         return 0;
      } else {
         erase(element);
         return 1;
      }
   }
    
   //Remove with tombstones on device
   __device__
   iterator erase(iterator keyPos){

      //Get the index of this entry
      size_t index = keyPos.getIndex();
      
      //If this is an empty bucket or a tombstone we can return already
      GID& item=buckets[index].first;
      if (item==EMPTYBUCKET || item==TOMBSTONE){return ++keyPos;}

      //Let's simply add a tombstone here
      atomicExch(&buckets[index].first,TOMBSTONE);
      atomicExch(&buckets[index].offset,0);
      atomicSub((unsigned int*)d_fill, 1);
      atomicAdd((unsigned int*)d_tombstoneCounter, 1);
      ++keyPos;
      return keyPos;
   }

   /**Device code for inserting elements. Nonexistent elements get created.
      Tombstones are accounted for.
    */
   __device__
   void insert_element(const GID& key,LID value, size_t &thread_overflowLookup) {
      int bitMask = (1 <<(*d_sizePower )) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);
      size_t i =0;
      while(i<buckets.size()){
         uint32_t vecindex=(hashIndex + i) & bitMask;
         GID old = atomicCAS(&buckets[vecindex].first, EMPTYBUCKET, key);
         //Key does not exist so we create it and incerement fill
         if (old == EMPTYBUCKET){
            atomicExch(&buckets[vecindex].first,key);
            atomicExch(&buckets[vecindex].second,value);
            atomicExch(&buckets[vecindex].offset,i);
            atomicAdd((unsigned int*)d_fill, 1);
            thread_overflowLookup = i+1;
            return;
         }

         //Key exists so we overwrite it. Fill stays the same
         if (old == key){
            atomicExch(&buckets[vecindex].second,value);
            thread_overflowLookup = i+1;
            return;
         }

         //Tombstone encounter. Fill gets inceremented and we look ahead for 
         //duplicates. If we find a duplicate we overwrite that otherwise we 
         //replace the tombstone with the new element
         //if (old==TOMBSTONE){
            //for (size_t j=i; j< thread_overflowLookup; j++){
               //uint32_t next_index=(hashIndex + j) & bitMask;
               //GID candidate;
               //atomicExch(&candidate,buckets[next_index].first);
               //if (candidate == key){
                  //atomicExch(&buckets[vecindex].second,value);
                  //thread_overflowLookup = i+1;
                  //return;
               //}
               //if (candidate == EMPTYBUCKET){
                  //atomicExch(&buckets[vecindex].first,key);
                  //atomicExch(&buckets[vecindex].second,value);
                  //atomicAdd((unsigned int*)d_fill, 1);
                  //thread_overflowLookup = i+1;
                  //return;
               //}
            //}

         //}
         i++;
      }
      assert(false && "Hashmap completely overflown");
   }

   __device__
   hash_pair<iterator, bool> insert(hash_pair<GID, LID> newEntry) {
      bool found = find(newEntry.first) != end();
      if (!found) {
         set_element(newEntry.first,newEntry.second);
      }
      return hash_pair<iterator, bool>(find(newEntry.first), !found);
   }

#endif

};

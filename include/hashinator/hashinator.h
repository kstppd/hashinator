/* File:    hasinator.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: A hybrid hashmap that can operate on both 
 *              CPUs and GPUs using CUDA unified memory.
 *
 * This file defines the following classes:
 *    --Hashinator::Hashmap;
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 * */
#pragma once
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <limits>
#include "../splitvector/splitvec.h"
#include "../splitvector/split_tools.h"
#include "hash_pair.h"
#include "defaults.h"
#include "hashfunctions.h"
#include "hashers.h"

namespace Hashinator{
   template <typename KEY_TYPE, 
             typename VAL_TYPE, 
             int maxBucketOverflow = 32, 
             KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
             KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1,
             class HashFunction=HashFunctions::Murmur<KEY_TYPE>,
             class DeviceHasher=Hashers::Hasher<KEY_TYPE,VAL_TYPE,HashFunction>>
   class Hashmap {

   private:
      //CUDA device handles
      int* d_sizePower;
      int* d_maxBucketOverflow;
      int postDevice_maxBucketOverflow;
      size_t* d_fill;
      size_t* d_tombstoneCounter;
      size_t tombstoneCounter;
      Hashmap* device_map;
      //~CUDA device handles

      //Host members
      int sizePower; // Logarithm (base two) of the size of the table
      int cpu_maxBucketOverflow;
      size_t fill;   // Number of filled buckets
      split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> buckets;
      //~Host members
      

       // Wrapper over available hash functions 
      __host__ __device__
      uint32_t hash(KEY_TYPE in) const {
          static_assert(std::is_arithmetic<KEY_TYPE>::value && sizeof(KEY_TYPE) <= sizeof(uint32_t));
          return HashFunction::_hash(in);
       }
      
      // Used by the constructors. Preallocates the device pointer and bookeepping info for later use on device. 
      // This helps in reducing the number of calls to cudaMalloc
      __host__
      void preallocate_device_handles(){
         cudaMalloc((void **)&d_sizePower, sizeof(int));
         cudaMalloc((void **)&d_maxBucketOverflow, sizeof(int));
         cudaMalloc((void **)&d_fill, sizeof(size_t));
         cudaMalloc((void **)&d_tombstoneCounter, sizeof(size_t));
         cudaMalloc((void **)&device_map, sizeof(Hashmap));
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


      //Cleans all tombstones using splitvectors stream compcation and
      //the member Hasher
      __host__
      void clean_tombstones(){
         //Reset the tomstone counter
         tombstoneCounter=0;
         //Allocate memory for overflown elements. So far this is the same size as our buckets but we can be better than this 
         //TODO size of overflown elements is known beforhand.
         split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> overflownElements(1 << sizePower, {EMPTYBUCKET, VAL_TYPE()});
         //Extract all overflown elements-This also resets TOMSBTONES to EMPTYBUCKET!
         split::tools::copy_if<hash_pair<KEY_TYPE, VAL_TYPE>,Overflown_Predicate<KEY_TYPE,VAL_TYPE>,defaults::WARPSIZE,defaults::WARPSIZE>(buckets,overflownElements,Overflown_Predicate<KEY_TYPE,VAL_TYPE>());
         size_t nOverflownElements=overflownElements.size();
         if (nOverflownElements ==0 ){
            std::cout<<"No cleaning needed!"<<std::endl;
            return ;
         }
         //If we do have overflown elements we put them back in the buckets
         Hashers::reset_to_empty<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction><<<overflownElements.size(),defaults::MAX_BLOCKSIZE>>> (overflownElements.data(),buckets.data(),sizePower,maxBucketOverflow,overflownElements.size());
         cudaDeviceSynchronize();
         DeviceHasher::insert(overflownElements.data(),buckets.data(),sizePower,maxBucketOverflow,d_maxBucketOverflow,d_fill,overflownElements.size());
         return ;
      }


   public:
      template <typename T, typename U>
      struct Overflown_Predicate{
         __host__ __device__
         inline bool operator()( hash_pair<T,U>& element)const{
            if (element.first==TOMBSTONE){element.first=EMPTYBUCKET;return false;}
            return element.offset>0;
         }
      };
      __host__
      Hashmap()
          : sizePower(5), fill(0), buckets(1 << sizePower, hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE())){
            preallocate_device_handles();
          };

      __host__
      Hashmap(int sizepower)
          : sizePower(sizepower), fill(0), buckets(1 << sizepower, hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE())){
            preallocate_device_handles();
          };
      __host__
      Hashmap(const Hashmap<KEY_TYPE, VAL_TYPE>& other)
          : sizePower(other.sizePower), fill(other.fill), buckets(other.buckets){
            preallocate_device_handles();
          };
      __host__
      ~Hashmap(){     
         deallocate_device_handles();
      };


      __host__
      void insert(hash_pair<KEY_TYPE,VAL_TYPE>*src,size_t len,int power){
         resize(power+1);
         cpu_maxBucketOverflow=maxBucketOverflow;
         cudaMemcpy(d_maxBucketOverflow,&cpu_maxBucketOverflow, sizeof(int),cudaMemcpyHostToDevice);
         cudaMemcpy(d_fill, &fill, sizeof(size_t),cudaMemcpyHostToDevice);
         DeviceHasher::insert(src,buckets.data(),sizePower,maxBucketOverflow,d_maxBucketOverflow,d_fill,len);
         cudaDeviceSynchronize();
         cudaMemcpyAsync(&fill, d_fill, sizeof(size_t),cudaMemcpyDeviceToHost,0);
         cudaMemcpyAsync(&cpu_maxBucketOverflow, d_maxBucketOverflow, sizeof(int),cudaMemcpyDeviceToHost,0);
         std::cout<<fill<<std::endl;
         std::cout<<cpu_maxBucketOverflow<<std::endl;
         if (cpu_maxBucketOverflow>maxBucketOverflow){
            std::cout<<"Rehashing..."<<std::endl;
            rehash(power+1);
         }
         return;
      }

      // Resize the table to fit more things. This is automatically invoked once
      // maxBucketOverflow has triggered. This can only be done on host (so far)
      __host__
      void rehash(int newSizePower) {
         if (newSizePower > 32) {
            throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
         }
         split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> newBuckets(1 << newSizePower,
                                                     hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
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
               hash_pair<KEY_TYPE, VAL_TYPE>& candidate = newBuckets[(newHash + i) & bitMask];
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
      VAL_TYPE& _at(const KEY_TYPE& key) {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
      const VAL_TYPE& _at(const KEY_TYPE& key) const {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
            if (candidate.first == key) {
               // Found a match, return that
               return candidate.second;
            }
            if (candidate.first == EMPTYBUCKET) {
               // Found an empty bucket, so error.
               throw std::out_of_range("Element not found in Hashmap.at");
            }
         }

         // Not found, so error.
         throw std::out_of_range("Element not found in Hashmap.at");
      }


      //---------------------------------------

      // For STL compatibility: size(), bucket_count(), count(KEY_TYPE), clear()
      __host__
      size_t size() const { return fill; }

      __host__ __device__
      size_t bucket_count() const {
         return buckets.size();
      }
      
      __host__
      float load_factor() const {return (float)size()/bucket_count();}

      __host__
      size_t count(const KEY_TYPE& key) const {
         if (find(key) != end()) {
            return 1;
         } else {
            return 0;
         }
      }

      __host__
      void clear() {
         buckets = split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>>(1 << sizePower, {EMPTYBUCKET, VAL_TYPE()});
         fill = 0;
      }

      //Try to grow our buckets until we achieve a targetLF load factor
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
       * Host function  that returns a device pointer that can be passed to CUDA kernels
       * The pointer is internally cleaned up by the destructors, however the user **must**
       * call download() after usage on device.
       */
      __host__
      Hashmap* upload(cudaStream_t stream = 0 ){
         cpu_maxBucketOverflow=maxBucketOverflow;
         this->buckets.optimizeGPU(stream); //already async so can be overlapped if used with streams
         cudaMemcpyAsync(d_sizePower, &sizePower, sizeof(int),cudaMemcpyHostToDevice,stream);
         cudaMemcpyAsync(d_maxBucketOverflow,&cpu_maxBucketOverflow, sizeof(int),cudaMemcpyHostToDevice,stream);
         cudaMemcpyAsync(d_fill, &fill, sizeof(size_t),cudaMemcpyHostToDevice,stream);
         cudaMemcpyAsync(device_map, this, sizeof(Hashmap),cudaMemcpyHostToDevice,stream);
         cudaMemsetAsync(d_tombstoneCounter, 0, sizeof(size_t)); //since tombstones do not exist on host code
         return device_map;
      }

      //Just return the device pointer. Upload should be called fist 
      //othewise map bookeepping info will not be updated on device.
      __host__
      Hashmap* get_device_pointer(){
         return device_map;
      }

      /**
       * This must be called after exiting a CUDA kernel. These functions
       * will do the following :
       *  • handle communicating bookeepping info back to host. 
       *  • If the hashmap has overflown on device it will try
       *  • If there are Tombstones then those are removed
       * */ 
      __host__
      void download(cudaStream_t stream = 0){
         //Copy over fill as it might have changed
         cudaMemcpyAsync(&fill, d_fill, sizeof(size_t),cudaMemcpyDeviceToHost,stream);
         cudaMemcpyAsync(&tombstoneCounter, d_tombstoneCounter, sizeof(size_t),cudaMemcpyDeviceToHost,stream);
         cudaMemcpyAsync(&postDevice_maxBucketOverflow, d_maxBucketOverflow, sizeof(int),cudaMemcpyDeviceToHost,stream);
         this->buckets.optimizeCPU(stream);
         if (postDevice_maxBucketOverflow>maxBucketOverflow){
            std::cout<<"Device Overflow"<<std::endl;
            rehash(sizePower+1);
         }else{
            if(tombstone_count()>0){
               std::cout<<"Cleaning Tombstones"<<std::endl;
               clean_tombstones();
            }
         }
      }

      __host__
         void print_pair(const hash_pair<KEY_TYPE, VAL_TYPE>& i)const {
            if (i.first==TOMBSTONE){
               std::cout<<"[╀,-,-] ";
            }else if (i.first == EMPTYBUCKET){
               std::cout<<"[▢,-,-] ";
            }
            else{
               if (i.offset>0){
                  printf("[%d,%d-\033[1;31m%d\033[0m]] ",i.first,i.second,i.offset);
               }else{
                  printf("[%d,%d-%d] ",i.first,i.second,i.offset);
               }
            }
         }
      __host__
      void dump_buckets()const {
         std::cout<<fill<<" "<<load_factor()<<std::endl;
         std::cout<<"\n";
         for  (int i =0 ; i < buckets.size(); ++i){
            print_pair(buckets[i]);
         }
         std::cout<<std::endl;

      }
       __host__
      size_t tombstone_count()const {
         return tombstoneCounter;
      }

      __host__
      void swap(Hashmap<KEY_TYPE, VAL_TYPE>& other) noexcept{
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
      const VAL_TYPE& at(const KEY_TYPE& key) const {
         return _at(key);
      }

      //See _at(key)
      __host__
      VAL_TYPE& at(const KEY_TYPE& key) {
         return _at(key);
      }

      // Typical array-like access with [] operator
      __host__
      VAL_TYPE& operator[](const KEY_TYPE& key) {
         return at(key); 
      }


      __device__
      void set_element(const KEY_TYPE& key,VAL_TYPE val){
         size_t thread_overflowLookup;
         insert_element(key,val,thread_overflowLookup);
         atomicMax(d_maxBucketOverflow,thread_overflowLookup);
      }

      __device__
      const VAL_TYPE& read_element(const KEY_TYPE& key) const {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < *d_maxBucketOverflow; i++) {
            uint32_t vecindex=(hashIndex + i) & bitMask;
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[vecindex];
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
      class iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<KEY_TYPE, VAL_TYPE>> {
         Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
         size_t index;

      public:
         __host__
         iterator(Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

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
         hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         __host__
         hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         __host__
         size_t getIndex() { return index; }
      };

      // Const iterator.
      class const_iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<KEY_TYPE, VAL_TYPE>> {
         const Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
         size_t index;

      public:
         __host__
         explicit const_iterator(const Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index)
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
         const hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         __host__
         const hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         __host__
         size_t getIndex() { return index; }
      };

      // Element access by iterator
      __host__
      const const_iterator find(KEY_TYPE key) const {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
      iterator find(KEY_TYPE key) {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
               KEY_TYPE nextBucket = buckets[(index + i)&bitMask].first;
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
                     VAL_TYPE moveValue = buckets[(index+i)&bitMask].second;
                     buckets[targetPos] = hash_pair<KEY_TYPE, VAL_TYPE>(nextBucket,moveValue);
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
      hash_pair<iterator, bool> insert(hash_pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = find(newEntry.first) != end();
         if (!found) {
            at(newEntry.first) = newEntry.second;
         }
         return hash_pair<iterator, bool>(find(newEntry.first), !found);
      }

      __host__
      size_t erase(const KEY_TYPE& key) {
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
      class iterator  {
      private:
         size_t index;
         Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
      public:
         __device__
         iterator(Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
         
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
         hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         __device__
         hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }

      };


      class const_iterator  {
      private:
         size_t index;
         const Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
      public:
         __device__
         explicit const_iterator(const Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
         
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
         const hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         __device__
         const hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
      };


      // Element access by iterator
      __device__ 
      iterator find(KEY_TYPE key) {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < *d_maxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

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
      const const_iterator find(KEY_TYPE key)const {
         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < *d_maxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

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
      size_t erase(const KEY_TYPE& key) {
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
         //TODO Use CAS here for safety
         KEY_TYPE& item=buckets[index].first;
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
      void insert_element(const KEY_TYPE& key,VAL_TYPE value, size_t &thread_overflowLookup) {
         int bitMask = (1 <<(*d_sizePower )) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);
         size_t i =0;
         while(i<buckets.size()){
            uint32_t vecindex=(hashIndex + i) & bitMask;
            KEY_TYPE old = atomicCAS(&buckets[vecindex].first, EMPTYBUCKET, key);
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
            if (old==TOMBSTONE){
               for (size_t j=i; j< thread_overflowLookup; j++){
                  uint32_t next_index=(hashIndex + j) & bitMask;
                  KEY_TYPE candidate;
                  atomicExch(&candidate,buckets[next_index].first);
                  if (candidate == key){
                     atomicExch(&buckets[vecindex].second,value);
                     thread_overflowLookup = i+1;
                     return;
                  }
                  if (candidate == EMPTYBUCKET){
                     atomicExch(&buckets[vecindex].first,key);
                     atomicExch(&buckets[vecindex].second,value);
                     atomicAdd((unsigned int*)d_fill, 1);
                     thread_overflowLookup = i+1;
                     return;
                  }
               }

            }
            i++;
         }
         assert(false && "Hashmap completely overflown");
      }

      __device__
      hash_pair<iterator, bool> insert(hash_pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = find(newEntry.first) != end();
         if (!found) {
            set_element(newEntry.first,newEntry.second);
         }
         return hash_pair<iterator, bool>(find(newEntry.first), !found);
      }

#endif

   };
}//namespace Hashinator

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


// Open bucket power-of-two sized hash table with multiplicative fibonacci hashing
template <typename GID, typename LID, int maxBucketOverflow = 32, GID EMPTYBUCKET = vmesh::INVALID_GLOBALID > 
class Hashinator {
private:
   int* d_sizePower;
   int* d_maxBucketOverflow;
   int postDevice_maxBucketOverflow;
   size_t* d_fill;
   int nBlocks=3;
   size_t cur_bucket_index=0;

   int sizePower; // Logarithm (base two) of the size of the table
   int cpu_maxBucketOverflow;
   size_t fill;   // Number of filled buckets
   split::SplitVector<split::SplitVector<std::pair<GID,LID>>> bucket_bank;
   split::SplitVector<std::pair<GID, LID>>* buckets;

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

    // Generic h
   __host__ __device__
   uint32_t hash(GID in) const {
       static constexpr bool n = (std::is_arithmetic<GID>::value && sizeof(GID) <= sizeof(uint32_t));

       if (n) {
          return fibonacci_hash(in);
       } else {
          return fnv_1a(&in, sizeof(GID));
       }
    }

public:
   Hashinator()
      : sizePower(4), fill(0){
         bucket_bank=split::SplitVector<split::SplitVector<std::pair<GID,LID>>>(nBlocks); 
         for (size_t i =0; i<nBlocks;i++){
            bucket_bank[i]=split::SplitVector<std::pair<GID,LID>>(1<<sizePower+i,std::pair<GID, LID>(EMPTYBUCKET, LID()));  
         }
         buckets=&bucket_bank[0];
      };
   Hashinator(const Hashinator<GID, LID>& other)
      : sizePower(other.sizePower), fill(other.fill), bucket_bank(other.bucket_bank){
         buckets=&bucket_bank[0];
      };

   // Resize the table to fit more things. This is automatically invoked once
   // maxBucketOverflow has triggered.
   void rehash(int newSizePower) {
#ifdef DEBUG
      std::cout<<"Rehashing to "<<( 1<<newSizePower )<<std::endl;
#endif
      if (newSizePower > 32) {
         throw std::out_of_range("Hashinator ran into rehashing catastrophe and exceeded 32bit buckets.");
      }
      split::SplitVector<std::pair<GID, LID>> newBuckets(1 << newSizePower,
                                                  std::pair<GID, LID>(EMPTYBUCKET, LID()));
      sizePower = newSizePower;
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size

      // Iterate through all old elements and rehash them into the new array.
      for (auto& e : bucket_bank[0]) {
         // Skip empty buckets
         if (e.first == EMPTYBUCKET) {
            continue;
         }

         uint32_t newHash = hash(e.first);
         bool found = false;
         for (int i = 0; i < maxBucketOverflow; i++) {
            std::pair<GID, LID>& candidate = newBuckets[(newHash + i) & bitMask];
            if (candidate.first == EMPTYBUCKET) {
               // Found an empty bucket, assign that one.
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
      bucket_bank[0] = newBuckets;
      buckets=&bucket_bank[0];
   }

   // Element access (by reference). Nonexistent elements get created.
   LID& at(const GID& key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         std::pair<GID, LID>& candidate = bucket_bank[0][(hashIndex + i) & bitMask];
         if (candidate.first == key) {
            // Found a match, return that
            return candidate.second;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, assign and return that.
            candidate.first = key;
            fill++;
            return candidate.second;
         }
      }

      // Not found, and we have no free slots to create a new one. So we need to rehash to a larger size.
      rehash(sizePower + 1);
      return at(key); // Recursive tail call to try again with larger table.
   }
      
   const LID& at(const GID& key) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const std::pair<GID, LID>& candidate = bucket_bank[0][(hashIndex + i) & bitMask];
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

   // Typical array-like access with [] operator
   LID& operator[](const GID& key) { return at(key); }

   // For STL compatibility: size(), bucket_count(), count(GID), clear()
   size_t size() const { return fill; }

   size_t bucket_count() const { return bucket_bank[0].size(); }
   
   float load_factor() const {return (float)size()/bucket_count();}

   size_t count(const GID& key) const {
      if (find(key) != end()) {
         return 1;
      } else {
         return 0;
      }
   }

   void clear() {
      std::cout<<"clearing"<<std::endl;
      for (size_t i =0; i<nBlocks;i++){
         bucket_bank[i]=split::SplitVector<std::pair<GID,LID>>(1<<sizePower+i,std::pair<GID, LID>(EMPTYBUCKET, LID()));  
      }
      buckets=bucket_bank.begin().data();
      fill = 0;
   }


   /************************Device Methods*********************************/
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

   __host__
   Hashinator* upload(){
      cpu_maxBucketOverflow=maxBucketOverflow;
      this->bucket_bank[0].optimizeGPU();
      Hashinator* device_map;
      cudaMalloc((void **)&d_sizePower, sizeof(int));
      cudaMemcpy(d_sizePower, &sizePower, sizeof(int),cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_maxBucketOverflow, sizeof(int));
      cudaMemcpy(d_maxBucketOverflow,&cpu_maxBucketOverflow, sizeof(int),cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_fill, sizeof(size_t));
      cudaMemcpy(d_fill, &fill, sizeof(size_t),cudaMemcpyHostToDevice);
      cudaMalloc((void **)&device_map, sizeof(Hashinator));
      cudaMemcpy(device_map, this, sizeof(Hashinator),cudaMemcpyHostToDevice);
      return device_map;
   }

   __host__
   void clean_up_after_device(Hashinator* device_map ){
      
      //Copy over fill as it might have changed
      cudaMemcpy(&fill, d_fill, sizeof(size_t),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&postDevice_maxBucketOverflow, d_maxBucketOverflow, sizeof(int),cudaMemcpyDeviceToHost);
      //TODO
      //Here postDevice_maxBucketOverflow is the new overflowing that took place while running on device
      //We need to handle this and clean up by rehashing to a larger container.
      std::cout<<"Overflow Limits Dev/Host "<<maxBucketOverflow<<"--> "<<postDevice_maxBucketOverflow<<std::endl;
      std::cout<<"Fill after device = "<<fill<<std::endl;
      this->bucket_bank[0].optimizeCPU();
      if (postDevice_maxBucketOverflow!=maxBucketOverflow){
         rehash(sizePower+1);
      }
     
      cudaFree(device_map);
      cudaFree(d_sizePower);
      cudaFree(d_maxBucketOverflow);
      cudaFree(d_fill);
   }

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

   void print_kvals(){
      for (auto it=begin(); it!=end();it++){
         std::cout<<it->first<<" "<<it->second<<std::endl;
      }
   }

   
   // Device code for element access (by reference). Nonexistent elements get created.
   __device__
   bool retrieve_w(const GID& key, size_t &thread_overflowLookup,LID* &retval) {
      int bitMask = (1 << *d_sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < thread_overflowLookup; i++) {
         uint32_t vecindex=(hashIndex + i) & bitMask;
         std::pair<GID, LID>& candidate = bucket_bank[0][vecindex];
         if (candidate.first == key) {
            // Found a match, return that
            retval = &candidate.second;
            return true;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, assign and return that.
            candidate.first = key;
            //compute capability 6.* and higher
            atomicAdd((unsigned long long  int*)d_fill, 1);
            retval = &candidate.second;
            return true;
         }
      }

      return false;
   }

   __device__
   bool retrieve_r(const GID& key,size_t thread_overflowLookup,LID* &retval) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         uint32_t vecindex=(hashIndex + i) & bitMask;
         const std::pair<GID, LID>& candidate = bucket_bank[0][vecindex];
         if (candidate.first == key) {
            // Found a match, return that
            retval = &candidate.second;
            return true;
         }
         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket, so error.
            return false;
         }
      }

      // Not found, so error.
      retval=nullptr;
      return false;
   }


   __device__
   LID* dev_at(const GID& key){
      size_t thread_overflowLookup=*d_maxBucketOverflow;
      LID* candidate=nullptr;
      bool found=false;
      while(!found){
         found = retrieve_w(key,thread_overflowLookup,candidate);
         if (!found){
            thread_overflowLookup+=1;
         }
         assert(thread_overflowLookup < bucket_bank[0].size() && "Buckets are completely overflown. This is a catastrophic failure...Consider .resize_to_lf()");
      }
      /*Now the local overflow might have changed for a thread. 
      We need to update the global one here.
      compute capability 3.5 and higher*/
      atomicMax(d_maxBucketOverflow,thread_overflowLookup);
      assert(candidate && "NULL pointer");
      return candidate;
   }


   __device__
   void set_element(const GID& key,LID val){
      LID* candidate= dev_at(key);
      static constexpr bool n = (std::is_arithmetic<LID>::value && sizeof(LID) <= sizeof(uint32_t));
      if(n){
         atomicExch((unsigned int*)candidate,(unsigned int)val);
      }else{
         atomicExch((unsigned long long int*)candidate,(unsigned long long int)val);
      }
   }

   __device__
   LID* read_element(const GID& key)const{
      LID* candidate=nullptr;
      bool found=false;
      size_t thread_overflowLookup=*d_maxBucketOverflow;
      found = retrieve_r(key,thread_overflowLookup,candidate);
      if (!found){
         assert(candidate && "Key element does not exist in the map");
      }
      return candidate;
   }

   /**************************device code*************************************************/

   // Iterator type. Iterates through all non-empty buckets.
   class iterator : public std::iterator<std::random_access_iterator_tag, std::pair<GID, LID>> {
      Hashinator<GID, LID>* hashtable;
      size_t index;

   public:
      iterator(Hashinator<GID, LID>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

      iterator& operator++() {
         index++;
         while(index < hashtable->bucket_bank[0].size()){
            if (hashtable->bucket_bank[0][index].first != EMPTYBUCKET){
               break;
            }
            index++;
         }
         return *this;
      }
      
      iterator operator++(int) { // Postfix version
         iterator temp = *this;
         ++(*this);
         return temp;
      }

      bool operator==(iterator other) const {
         return &hashtable->bucket_bank[0][index] == &other.hashtable->bucket_bank[0][other.index];
      }
      bool operator!=(iterator other) const {
         return &hashtable->bucket_bank[0][index] != &other.hashtable->bucket_bank[0][other.index];
      }
      std::pair<GID, LID>& operator*() const { return hashtable->bucket_bank[0][index]; }
      std::pair<GID, LID>* operator->() const { return &hashtable->bucket_bank[0][index]; }
      size_t getIndex() { return index; }
   };

   // Const iterator.
   class const_iterator : public std::iterator<std::random_access_iterator_tag, std::pair<GID, LID>> {
      const Hashinator<GID, LID>* hashtable;
      size_t index;

   public:
      explicit const_iterator(const Hashinator<GID, LID>& hashtable, size_t index)
          : hashtable(&hashtable), index(index) {}

      const_iterator& operator++() {
         index++;
         while(index < hashtable->bucket_bank[0].size()){
            if (hashtable->bucket_bank[0][index].first != EMPTYBUCKET){
               break;
            }
            index++;
         }
         return *this;
      }
      const_iterator operator++(int) { // Postfix version
         const_iterator temp = *this;
         ++(*this);
         return temp;
      }

      bool operator==(const_iterator other) const {
         return &hashtable->bucket_bank[0][index] == &other.hashtable->bucket_bank[0][other.index];
      }
      bool operator!=(const_iterator other) const {
         return &hashtable->bucket_bank[0][index] != &other.hashtable->bucket_bank[0][other.index];
      }
      const std::pair<GID, LID>& operator*() const { return hashtable->bucket_bank[0][index]; }
      const std::pair<GID, LID>* operator->() const { return &hashtable->bucket_bank[0][index]; }
      size_t getIndex() { return index; }
   };

   iterator begin() {
      for (size_t i = 0; i < bucket_bank[0].size(); i++) {
         if (bucket_bank[0][i].first != EMPTYBUCKET) {
            return iterator(*this, i);
         }
      }
      return end();
   }
   const_iterator begin() const {
      for (size_t i = 0; i < bucket_bank[0].size(); i++) {
         if (bucket_bank[0][i].first != EMPTYBUCKET) {
            return const_iterator(*this, i);
         }
      }
      return end();
   }

   iterator end() { return iterator(*this, bucket_bank[0].size()); }
   const_iterator end() const { return const_iterator(*this, bucket_bank[0].size()); }

   // Element access by iterator
   iterator find(GID key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const std::pair<GID, LID>& candidate = bucket_bank[0][(hashIndex + i) & bitMask];
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

   const const_iterator find(GID key) const {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const std::pair<GID, LID>& candidate = bucket_bank[0][(hashIndex + i) & bitMask];
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

   // More STL compatibility implementations
   std::pair<iterator, bool> insert(std::pair<GID, LID> newEntry) {
      bool found = find(newEntry.first) != end();
      if (!found) {
         at(newEntry.first) = newEntry.second;
      }
      return std::pair<iterator, bool>(find(newEntry.first), !found);
   }

   // Remove one element from the hash table.
   iterator erase(iterator keyPos) {
      // Due to overflowing buckets, this might require moving quite a bit of stuff around.
      size_t index = keyPos.getIndex();

      if (bucket_bank[0][index].first != EMPTYBUCKET) {
         // Decrease fill count
         fill--;

         // Clear the element itself.
         bucket_bank[0][index].first = EMPTYBUCKET;

         int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
         size_t targetPos = index;
         // Search ahead to verify items are in correct places (until empty bucket is found)
         for (unsigned int i = 1; i < fill; i++) {
            GID nextBucket = bucket_bank[0][(index + i)&bitMask].first;
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
                  LID moveValue = bucket_bank[0][(index+i)&bitMask].second;
                  bucket_bank[0][targetPos] = std::pair<GID, LID>(nextBucket,moveValue);
                  targetPos = ((index+i)&bitMask);
                  bucket_bank[0][targetPos].first = EMPTYBUCKET;
               }
            }
         }
      }
      // return the next valid bucket member
      ++keyPos;
      return keyPos;
   }
   size_t erase(const GID& key) {
      iterator element = find(key);
      if(element == end()) {
         return 0;
      } else {
         erase(element);
         return 1;
      }
   }

   void swap(Hashinator<GID, LID>& other) {
      std::cout<<"SWAPPING"<<std::endl;
      bucket_bank.swap(other.bucket_bank);
      int tempSizePower = sizePower;
      sizePower = other.sizePower;
      other.sizePower = tempSizePower;

      size_t tempFill = fill;
      fill = other.fill;
      other.fill = tempFill;

      int target_capacity=1<<sizePower;
      for (auto b=bucket_bank.begin();b!=bucket_bank.end();++b){
         if (b.data()->size()==target_capacity){
            buckets=b.data();
            break;
         }
      }

      int other_target_capacity=1<<other.sizePower;
      for (auto b=other.bucket_bank.begin();b!=other.bucket_bank.end();++b){
         if (b.data()->size()==other_target_capacity){
            other.buckets=b.data();
            break;
         }
      }
   }
};

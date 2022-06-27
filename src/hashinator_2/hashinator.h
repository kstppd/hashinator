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
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "definitions.h"
#include <stdlib.h>
#include "../splitvector/splitvec.h"

enum POLICY{
   ECO,
   COMFORT,
   PERFORMANCE,
   MEMHOG
};

// Open bucket power-of-two sized hash table with multiplicative fibonacci hashing
template <typename GID, typename LID, int maxBucketOverflow = 4, GID EMPTYBUCKET = vmesh::INVALID_GLOBALID >
 class Hashinator {
private:
   int sizePower; // Logarithm (base two) of the size of the table
   size_t fill;   // Number of filled buckets
   int nBlocks=4;                  
   split::SplitVector<std::pair<GID, LID>>* buckets;
   split::SplitVector<split::SplitVector<std::pair<GID,LID>>> bucket_bank;
   split::split_iterator<split::SplitVector<split::SplitVector<std::pair<GID,LID>>>> bucket_iter;
   // Fibonacci hash function for 64bit values
   uint32_t fibonacci_hash(GID in) const {
      in ^= in >> (32 - sizePower);
      uint32_t retval = (uint64_t)(in * 2654435769ul) >> (32 - sizePower);
      return retval;
   }

    //Hash a chunk of memory using fnv_1a
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
         buckets=bucket_bank.begin().data();
      }; 

   Hashinator(const Hashinator<GID, LID>& other) 
      : sizePower(other.sizePower), fill(other.fill),bucket_bank(other.bucket_bank){
      };

   // Resize the table to fit more things. This is automatically invoked once
   // maxBucketOverflow has triggered.
   void rehash(int newSizePower) {
      if (newSizePower > 32) {
         throw std::out_of_range("Hashinator ran into rehashing catastrophe and exceeded 32bit buckets.");
      }
#ifdef DEBUG
      std::cout<<"Rehashing to "<<( 1<<newSizePower )<<std::endl;
#endif
      split::SplitVector<std::pair<GID, LID>> newBucket_list(1 << newSizePower,
                                                  std::pair<GID, LID>(EMPTYBUCKET, LID()));
      sizePower = newSizePower;
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size

      // Iterate through all old elements and rehash them into the new array.
      for (auto& e : *buckets) {
         // Skip empty buckets
         if (e.first == EMPTYBUCKET) {
            continue;
         }

         uint32_t newHash = hash(e.first);
         bool found = false;
         for (int i = 0; i < maxBucketOverflow; i++) {
            std::pair<GID, LID>& candidate = newBucket_list[(newHash + i) & bitMask];
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
      *buckets = newBucket_list;
   }

   // Element access (by reference). Nonexistent elements get created.
   LID& at(const GID& key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         std::pair<GID, LID>& candidate = buckets->operator[]((hashIndex + i) & bitMask);
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
         const std::pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
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
   /*****************Device Code***********************************************/
   __host__
   void print_all(){
      std::cout<<">>>>*********************************"<<std::endl;
      std::cout<<"Map contains "<<bucket_count()<<" buckets"<<std::endl;
      std::cout<<"Map fill is "<<fill<<std::endl;
      std::cout<<"Map size is "<<size()<<std::endl;
      std::cout<<"Map LF is "<<load_factor()<<std::endl;
      std::cout<<"<<<<*********************************"<<std::endl;
   }

   void print_kvals(){
      for (auto it=begin(); it!=end();it++){
         std::cout<<it->first<<" "<<it->second<<std::endl;
      }
   }

   void print_bank(){
      for (auto& vec:bucket_bank ){
         printf("Bucket bank with size %zu\n",vec.size());
      }
   }
   /***********************************************************************/

   // Typical array-like access with [] operator
   LID& operator[](const GID& key) { return at(key); }

   // For STL compatibility: size(), bucket_count(), count(GID), clear()
   size_t size() const { return fill; }

   size_t bucket_count() const { return buckets->size(); }

   float load_factor() const {return (float)size()/bucket_count();}

   size_t count(const GID& key) const {
      if (find(key) != end()) {
         return 1;
      } else {
         return 0;
      }
   }

   void clear() {
      buckets = split::SplitVector<std::pair<GID, LID>>(1 << sizePower, {EMPTYBUCKET, LID()});
      fill = 0;
   }

   // Iterator type. Iterates through all non-empty buckets.
   class iterator : public std::iterator<std::random_access_iterator_tag, std::pair<GID, LID>> {
      Hashinator<GID, LID>* hashtable;
      size_t index;

   public:
      iterator(Hashinator<GID, LID>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

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
      
      iterator operator++(int) { // Postfix version
         iterator temp = *this;
         ++(*this);
         return temp;
      }

      bool operator==(iterator other) const {
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      bool operator!=(iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      std::pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      std::pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }
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
         while(index < hashtable->buckets.size()){
            if (hashtable->buckets[index].first != EMPTYBUCKET){
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
         return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
      }
      bool operator!=(const_iterator other) const {
         return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
      }
      const std::pair<GID, LID>& operator*() const { return hashtable->buckets[index]; }
      const std::pair<GID, LID>* operator->() const { return &hashtable->buckets[index]; }
      size_t getIndex() { return index; }
   };

   iterator begin() {
      for (size_t i = 0; i < buckets->size(); i++) {
         if (buckets[i].first != EMPTYBUCKET) {
            return iterator(*this, i);
         }
      }
      return end();
   }
   const_iterator begin() const {
      for (size_t i = 0; i < buckets->size(); i++) {
         if (buckets[i].first != EMPTYBUCKET) {
            return const_iterator(*this, i);
         }
      }
      return end();
   }

   iterator end() { return iterator(*this, buckets->size()); }
   const_iterator end() const { return const_iterator(*this, buckets->size()); }

   // Element access by iterator
   iterator find(GID key) {
      int bitMask = (1 << sizePower) - 1; // For efficient modulo of the array size
      uint32_t hashIndex = hash(key);

      // Try to find the matching bucket.
      for (int i = 0; i < maxBucketOverflow; i++) {
         const std::pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
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
         const std::pair<GID, LID>& candidate = buckets[(hashIndex + i) & bitMask];
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
                  buckets[targetPos] = std::pair<GID, LID>(nextBucket,moveValue);
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
      buckets->swap(other.buckets);
      int tempSizePower = sizePower;
      sizePower = other.sizePower;
      other.sizePower = tempSizePower;

      size_t tempFill = fill;
      fill = other.fill;
      other.fill = tempFill;
   }
};


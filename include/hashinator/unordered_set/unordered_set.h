#pragma once
/* File:    unordered_set.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 *
 * This file defines the following classes:
 *    --Hashinator::Unordered_Set;
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
#ifdef HASHINATOR_CPU_ONLY_MODE
#define SPLIT_CPU_ONLY_MODE
#endif
#include "../../common.h"
#include "../../splitvector/gpu_wrappers.h"
#include "../../splitvector/split_allocators.h"
#include "../../splitvector/splitvec.h"
#include "../defaults.h"
#include "../hashfunctions.h"
#include <algorithm>
#include <stdexcept>
#ifndef HASHINATOR_CPU_ONLY_MODE
#include "../../splitvector/split_tools.h"
#include "../hashers.h"
#endif

#define UNUSED(x) (void)(x)
namespace Hashinator {

#ifndef HASHINATOR_CPU_ONLY_MODE
template <typename T>
using DefaultMetaAllocator = split::split_unified_allocator<T>;
#define DefaultHasher                                                                                                  \
   Hashers::Hasher<KEY_TYPE, VAL_TYPE, HashFunction, EMPTYBUCKET, TOMBSTONE, defaults::WARPSIZE,                       \
                   defaults::elementsPerWarp>
#else
template <typename T>
using DefaultMetaAllocator = split::split_host_allocator<T>;
#define DefaultHasher void
#endif

typedef struct Info {
   Info(){};
   Info(int sz)
       : sizePower(sz), fill(0), currentMaxBucketOverflow(defaults::BUCKET_OVERFLOW), tombstoneCounter(0),
         err(status::invalid) {}
   int sizePower;
   size_t fill;
   size_t currentMaxBucketOverflow;
   size_t tombstoneCounter;
   status err;
} SetInfo;

template <typename KEY_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1, class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>,
          class Meta_Allocator = DefaultMetaAllocator<SetInfo>>

class Unordered_Set {

   // members
private:
   Unordered_Set* device_set;
   split::SplitVector<KEY_TYPE> buckets;
   Meta_Allocator _metaAllocator;
   SetInfo* _setInfo;

   HASHINATOR_HOSTDEVICE
   uint32_t hash(KEY_TYPE in) const {
      static_assert(std::is_arithmetic<KEY_TYPE>::value);
      return HashFunction::_hash(in, _setInfo->sizePower);
   }

   void addKey(KEY_TYPE key) noexcept {
      int bitMask = (1 << _setInfo->sizePower) - 1;
      auto hashIndex = hash(key);
      // Try to find the matching bucket.
      for (size_t i = 0; i < _setInfo->currentMaxBucketOverflow; i++) {
         KEY_TYPE& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate == EMPTYBUCKET) {
            candidate = key;
            _setInfo->fill++;
            return;
         }
         if (candidate == TOMBSTONE) {
            continue;
         }
      }
      rehash(_setInfo->sizePower + 1);
      return addKey(key);
   }

public:
   // Constructors Destructors and = Operators with move/cpy semantics
   Unordered_Set(uint32_t sizePower = 5) {
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(sizePower);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
   }

   Unordered_Set(const Unordered_Set& other) {
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = *other._setInfo;
      buckets = other.buckets;
   }

   Unordered_Set(const std::initializer_list<KEY_TYPE>& list) {
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(5);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
      for (size_t i = 0; i < list.size(); i++) {
         insert(list.begin()[i]);
      }
   }

   Unordered_Set(std::initializer_list<KEY_TYPE>&& list) {
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(5);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
      for (size_t i = 0; i < list.size(); i++) {
         insert(list.begin()[i]);
      }
   }

   Unordered_Set(Unordered_Set&& other) noexcept {
      *_setInfo = other.SetInfo;
      other._setInfo = nullptr;
      buckets = std::move(other.buckets);
   }

   ~Unordered_Set() { _metaAllocator.deallocate(_setInfo, 1); }

   Unordered_Set& operator=(const Unordered_Set& other) {
      if (this == &other) {
         return *this;
      }
      *_setInfo = *(other._setInfo);
      buckets = other.buckets;
      return *this;
   }

   Unordered_Set& operator=(Unordered_Set&& other) noexcept {
      if (this == &other) {
         return *this;
      }
      _metaAllocator.deallocate(_setInfo, 1);
      _setInfo = other._setInfo;
      other._setInfo = nullptr;
      buckets = std::move(other.buckets);
      return *this;
   }

   void rehash(uint32_t newSizePower) {
      if (newSizePower > 32) {
         throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
      }
      split::SplitVector<KEY_TYPE> newBuckets(1 << newSizePower, EMPTYBUCKET);
      _setInfo->sizePower = newSizePower;
      int bitMask = (1 << _setInfo->sizePower) - 1; // For efficient modulo of the array size

      // Iterate through all old elements and rehash them into the new array.
      for (auto& e : buckets) {
         // Skip empty buckets ; We also check for TOMBSTONE elements
         // as we might be coming off a kernel that overflew the hashmap
         if (e == EMPTYBUCKET || e == TOMBSTONE) {
            continue;
         }

         uint32_t newHash = hash(e);
         bool found = false;
         for (int i = 0; i < Hashinator::defaults::BUCKET_OVERFLOW; i++) {
            KEY_TYPE& candidate = newBuckets[(newHash + i) & bitMask];
            if (candidate == EMPTYBUCKET) {
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
      buckets = newBuckets;
      _setInfo->currentMaxBucketOverflow = Hashinator::defaults::BUCKET_OVERFLOW;
      _setInfo->tombstoneCounter = 0;
   }

   void rehash() { rehash(_setInfo->sizePower); }

   // Iterators
   class iterator {
      Unordered_Set<KEY_TYPE>* set;
      size_t index;

   public:
      iterator(Unordered_Set<KEY_TYPE>& set, size_t index) : set(&set), index(index) {}

      iterator& operator++() {
         index++;
         while (index < set->buckets.size()) {
            if (set->buckets[index] != EMPTYBUCKET && set->buckets[index] != TOMBSTONE) {
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
      bool operator==(iterator other) const { return &set->buckets[index] == &other.set->buckets[other.index]; }
      bool operator!=(iterator other) const { return &set->buckets[index] != &other.set->buckets[other.index]; }
      KEY_TYPE& operator*() const { return set->buckets[index]; }
      KEY_TYPE* operator->() const { return &set->buckets[index]; }
      size_t getIndex() { return index; }
   };

   // Const iterator.
   class const_iterator {
      const Unordered_Set<KEY_TYPE>* set;
      size_t index;

   public:
      explicit const_iterator(const Unordered_Set<KEY_TYPE>& set, size_t index) : set(&set), index(index) {}
      const_iterator& operator++() {
         index++;
         while (index < set->buckets.size()) {
            if (set->buckets[index] != EMPTYBUCKET && set->buckets[index] != TOMBSTONE) {
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
      bool operator==(const_iterator other) const { return &set->buckets[index] == &other.set->buckets[other.index]; }
      bool operator!=(const_iterator other) const { return &set->buckets[index] != &other.set->buckets[other.index]; }
      const KEY_TYPE& operator*() const { return set->buckets[index]; }
      const KEY_TYPE* operator->() const { return &set->buckets[index]; }
      size_t getIndex() { return index; }
   };

   // Device Iterator type. Iterates through all non-empty buckets.
   class device_iterator {
   private:
      size_t index;
      Unordered_Set<KEY_TYPE>* set;

   public:
      HASHINATOR_DEVICEONLY
      device_iterator(Unordered_Set<KEY_TYPE>& set, size_t index) : index(index), set(&set) {}

      HASHINATOR_DEVICEONLY
      size_t getIndex() { return index; }

      HASHINATOR_DEVICEONLY
      device_iterator& operator++() {
         index++;
         while (index < set->buckets.size()) {
            if (set->buckets[index] != EMPTYBUCKET && set->buckets[index] != TOMBSTONE) {
               break;
            }
            index++;
         }
         return *this;
      }

      HASHINATOR_DEVICEONLY
      device_iterator operator++(int) {
         device_iterator temp = *this;
         ++(*this);
         return temp;
      }

      HASHINATOR_DEVICEONLY
      bool operator==(device_iterator other) const { return &set->buckets[index] == &other.set->buckets[other.index]; }
      HASHINATOR_DEVICEONLY
      bool operator!=(device_iterator other) const { return &set->buckets[index] != &other.set->buckets[other.index]; }

      HASHINATOR_DEVICEONLY
      KEY_TYPE& operator*() const { return set->buckets[index]; }
      HASHINATOR_DEVICEONLY
      KEY_TYPE* operator->() const { return &set->buckets[index]; }
   };

   class const_device_iterator {
   private:
      size_t index;
      const Unordered_Set<KEY_TYPE>* set;

   public:
      HASHINATOR_DEVICEONLY
      explicit const_device_iterator(const Unordered_Set<KEY_TYPE>& set, size_t index) : index(index), set(&set) {}

      HASHINATOR_DEVICEONLY
      size_t getIndex() { return index; }

      HASHINATOR_DEVICEONLY
      const_device_iterator& operator++() {
         index++;
         while (index < set->buckets.size()) {
            if (set->buckets[index] != EMPTYBUCKET && set->buckets[index] != TOMBSTONE) {
               break;
            }
            index++;
         }
         return *this;
      }

      HASHINATOR_DEVICEONLY
      const_device_iterator operator++(int) {
         const_device_iterator temp = *this;
         ++(*this);
         return temp;
      }

      HASHINATOR_DEVICEONLY
      bool operator==(const_device_iterator other) const {
         return &set->buckets[index] == &other.set->buckets[other.index];
      }
      HASHINATOR_DEVICEONLY
      bool operator!=(const_device_iterator other) const {
         return &set->buckets[index] != &other.set->buckets[other.index];
      }

      HASHINATOR_DEVICEONLY
      const KEY_TYPE& operator*() const { return set->buckets[index]; }
      HASHINATOR_DEVICEONLY
      const KEY_TYPE* operator->() const { return &set->buckets[index]; }
   };

   iterator begin() {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i] != EMPTYBUCKET && buckets[i] != TOMBSTONE) {
            return iterator(*this, i);
         }
      }
      return end();
   }

   const_iterator begin() const {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i] != EMPTYBUCKET && buckets[i] != TOMBSTONE) {
            return const_iterator(*this, i);
         }
      }
      return end();
   }

   iterator end() { return iterator(*this, buckets.size()); }

   const_iterator end() const { return const_iterator(*this, buckets.size()); }

   HASHINATOR_DEVICEONLY
   device_iterator device_end() { return device_iterator(*this, buckets.size()); }

   HASHINATOR_DEVICEONLY
   const_device_iterator device_end() const { return const_device_iterator(*this, buckets.size()); }

   HASHINATOR_DEVICEONLY
   device_iterator device_begin() {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i] != EMPTYBUCKET && buckets[i] != TOMBSTONE) {
            return device_iterator(*this, i);
         }
      }
      return device_end();
   }

   HASHINATOR_DEVICEONLY
   const_device_iterator device_begin() const {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i] != EMPTYBUCKET && buckets[i] != TOMBSTONE) {
            return const_device_iterator(*this, i);
         }
      }
      return device_end();
   }

   void print_pair(const KEY_TYPE& i) const {
      size_t currentSizePower = _setInfo->sizePower;
      const size_t hashIndex = HashFunction::_hash(i, currentSizePower);
      const int bitMask = (1 << (currentSizePower)) - 1;
      size_t optimalIndex = hashIndex & bitMask;
      const_iterator it = find(i);
      int64_t overflow = llabs(it.getIndex() - optimalIndex);
      if (i == TOMBSTONE) {
         std::cout << "[╀] ";
      } else if (i == EMPTYBUCKET) {
         std::cout << "[▢] ";
      } else {
         if (overflow > 0) {
            printf("[%d,\033[1;31m%li\033[0m] ", i, overflow);
         } else {
            printf("[%d,%zu] ", i, overflow);
         }
      }
   }

   void dump_buckets() const {
      printf("Hashinator Stats \n");
      printf("Fill= %zu, LoadFactor=%f \n", _setInfo->fill, load_factor());
      printf("Tombstones= %zu\n", _setInfo->tombstoneCounter);
      for (size_t i = 0; i < buckets.size(); ++i) {
         print_pair(buckets[i]);
      }
      printf("\n");
   }

   HASHINATOR_HOSTDEVICE
   void stats() const {
      printf("Hashinator Stats \n");
      printf("Bucket size= %lu\n", buckets.size());
      printf("Fill= %lu, LoadFactor=%f \n", _setInfo->fill, load_factor());
      printf("Tombstones= %lu\n", _setInfo->tombstoneCounter);
      printf("Overflow= %lu\n", _setInfo->currentMaxBucketOverflow);
   }

   HASHINATOR_HOSTDEVICE
   inline int getSizePower(void) const noexcept { return _setInfo->sizePower; }

   // For STL compatibility: size(), bucket_count(), count(KEY_TYPE), clear()
   HASHINATOR_HOSTDEVICE
   size_t size() const noexcept { return _setInfo->fill; }

   HASHINATOR_HOSTDEVICE
   size_t bucket_count() const noexcept { return buckets.size(); }

   HASHINATOR_HOSTDEVICE
   float load_factor() const noexcept { return (float)size() / bucket_count(); }

   HASHINATOR_HOSTDEVICE
   size_t tombstone_count() const noexcept { return _setInfo->tombstoneCounter; }

   bool contains(const KEY_TYPE& key) const noexcept { return (find(key) == end()) ? false : true; }

   bool empty() const noexcept { return begin() == end(); }

   size_t count(const KEY_TYPE& key) const noexcept { return contains(key) ? 1 : 0; }

   iterator find(const KEY_TYPE& key) noexcept {
      const int bitMask = (1 << _setInfo->sizePower) - 1;
      const auto hashIndex = hash(key);

      for (size_t i = 0; i < _setInfo->currentMaxBucketOverflow; i++) {
         const KEY_TYPE& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate == key) {
            auto index = (hashIndex + i) & bitMask;
            return iterator(*this, index);
         }
         if (candidate == TOMBSTONE) {
            continue;
         }
         if (candidate == EMPTYBUCKET) {
            return end();
         }
      }
      return end();
   }

   const const_iterator find(const KEY_TYPE& key) const noexcept {
      const int bitMask = (1 << _setInfo->sizePower) - 1;
      const auto hashIndex = hash(key);

      for (size_t i = 0; i < _setInfo->currentMaxBucketOverflow; i++) {
         const KEY_TYPE& candidate = buckets[(hashIndex + i) & bitMask];
         if (candidate == key) {
            auto index = (hashIndex + i) & bitMask;
            return const_iterator(*this, index);
         }
         if (candidate == TOMBSTONE) {
            continue;
         }
         if (candidate == EMPTYBUCKET) {
            return end();
         }
      }
      return end();
   }

   hash_pair<iterator, bool> insert(const KEY_TYPE& key) noexcept {
      // try to find key
      iterator it = find(key);

      // if the key already exists we mutate it
      if (it != end()) {
         *it = key;
         return {it, it != end()};
      }
      // otherwise we add it
      addKey(key);
      iterator retval = find(key);
      return {retval, retval != end()};
   }

   hash_pair<iterator, bool> insert(KEY_TYPE&& key) noexcept {
      // try to find key
      iterator it = find(key);

      // if the key already exists we mutate it
      if (it != end()) {
         *it = key;
         return {it, it != end()};
      }
      // otherwise we add it
      addKey(key);
      iterator retval = find(key);
      return {retval, retval != end()};
   }

   iterator erase(iterator pos) {
      auto index = pos.getIndex();
      assert(index < 1 << _setInfo->sizePower);
      KEY_TYPE& key = buckets[index];
      if (key != EMPTYBUCKET && key != TOMBSTONE) {
         key = TOMBSTONE;
         _setInfo->fill--;
         _setInfo->tombstoneCounter++;
      }
      return ++pos; // return next valid element;
   }

   iterator erase(const_iterator pos) {
      auto index = pos.getIndex();
      assert(index < 1 << _setInfo->sizePower);
      KEY_TYPE& key = buckets[index];
      if (key != EMPTYBUCKET && key != TOMBSTONE) {
         key = TOMBSTONE;
         _setInfo->fill--;
         _setInfo->tombstoneCounter++;
      }
      return ++pos; // return next valid element;
   }
   

   void insert_fast(const KEY_TYPE* keys, size_t len){

   }



}; // Unordered_Set

} // namespace Hashinator

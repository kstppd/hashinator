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
#include "../hash_pair.h"
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
   Hashers::Hasher<KEY_TYPE, void, HashFunction, EMPTYBUCKET, TOMBSTONE, defaults::WARPSIZE,                       \
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
          class DeviceHasher = DefaultHasher, class Meta_Allocator = DefaultMetaAllocator<SetInfo>>

class Unordered_Set {

   // members
private:
   SetInfo* _setInfo;
   split::SplitVector<KEY_TYPE> buckets;
   Meta_Allocator _metaAllocator;
   Unordered_Set* device_set;

   HASHINATOR_HOSTDEVICE
   uint32_t hash(KEY_TYPE in) const {
      static_assert(std::is_arithmetic<KEY_TYPE>::value);
      return HashFunction::_hash(in, _setInfo->sizePower);
   }

   HASHINATOR_HOSTDEVICE
   inline void set_status(status code) noexcept { _setInfo->err = code; }

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

   void preallocate_device_handles() {
   #ifndef HASHINATOR_CPU_ONLY_MODE
      SPLIT_CHECK_ERR(split_gpuMalloc((void**)&device_set, sizeof(Unordered_Set)));
   #endif
   }

   // Deallocates the bookeepping info and the device pointer
   void deallocate_device_handles() {
      if (device_set == nullptr) {
         return;
      }
   #ifndef HASHINATOR_CPU_ONLY_MODE
      SPLIT_CHECK_ERR(split_gpuFree(device_set));
      device_set= nullptr;
   #endif
   }

public:
   // Constructors Destructors and = Operators with move/cpy semantics
   Unordered_Set(uint32_t sizePower = 5) {
      preallocate_device_handles();
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(sizePower);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
   }

   Unordered_Set(const Unordered_Set& other) {
      preallocate_device_handles();
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = *other._setInfo;
      buckets = other.buckets;
   }

   Unordered_Set(const std::initializer_list<KEY_TYPE>& list) {
      preallocate_device_handles();
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(5);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
      for (size_t i = 0; i < list.size(); i++) {
         insert(list.begin()[i]);
      }
   }

   Unordered_Set(const std::vector<KEY_TYPE>& vec) {
      preallocate_device_handles();
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(5);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
      for (size_t i = 0; i < vec.size(); i++) {
         insert(vec.begin()[i]);
      }
   }

   Unordered_Set(std::initializer_list<KEY_TYPE>&& list) {
      preallocate_device_handles();
      _setInfo = _metaAllocator.allocate(1);
      *_setInfo = SetInfo(5);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, EMPTYBUCKET);
      for (size_t i = 0; i < list.size(); i++) {
         insert(list.begin()[i]);
      }
   }

   Unordered_Set(Unordered_Set&& other) noexcept {
      preallocate_device_handles();
      _setInfo = other._setInfo;
      other._setInfo = nullptr;
      buckets = std::move(other.buckets);
   }

   ~Unordered_Set() { 
      deallocate_device_handles();
      _metaAllocator.deallocate(_setInfo, 1); 
   }

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

   HASHINATOR_HOSTDEVICE
   inline status peek_status(void) noexcept {
      status retval = _setInfo->err;
      _setInfo->err = status::invalid;
      return retval;
   }

#ifdef HASHINATOR_CPU_ONLY_MODE
   void* operator new(size_t len) {
      void* ptr = (void*)malloc(len);
      return ptr;
   }

   void operator delete(void* ptr) { free(ptr); }

   void* operator new[](size_t len) {
      void* ptr = (void*)malloc(len);
      return ptr;
   }

   void operator delete[](void* ptr) { free(ptr); }

#else
   void* operator new(size_t len) {
      void* ptr;
      SPLIT_CHECK_ERR(split_gpuMallocManaged(&ptr, len));
      return ptr;
   }

   void operator delete(void* ptr) { SPLIT_CHECK_ERR(split_gpuFree(ptr)); }

   void* operator new[](size_t len) {
      void* ptr;
      SPLIT_CHECK_ERR(split_gpuMallocManaged(&ptr, len));
      return ptr;
   }

   void operator delete[](void* ptr) { split_gpuFree(ptr); }

   void copyMetadata(SetInfo* dst, split_gpuStream_t s = 0) {
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(dst, _setInfo, sizeof(SetInfo), split_gpuMemcpyDeviceToHost, s));
   }

#endif

#ifndef HASHINATOR_CPU_ONLY_MODE
   Unordered_Set* upload(split_gpuStream_t stream = 0) {
      optimizeGPU(stream);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(device_set, this, sizeof(Unordered_Set), split_gpuMemcpyHostToDevice, stream));
      return device_set;
   }

   void download(split_gpuStream_t stream = 0) {
      // Copy over fill as it might have changed
      optimizeCPU(stream);
      if (_setInfo->currentMaxBucketOverflow > Hashinator::defaults::BUCKET_OVERFLOW) {
         rehash(_setInfo->sizePower + 1);
      } else {
         if (tombstone_count() > 0) {
            clean_tombstones(stream);
         }
      }
   }

   void optimizeGPU(split_gpuStream_t stream = 0) noexcept {
      int device;
      SPLIT_CHECK_ERR(split_gpuGetDevice(&device));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_setInfo, sizeof(SetInfo), device, stream));
      buckets.optimizeGPU(stream);
   }

   /*Manually prefetch data on Host*/
   void optimizeCPU(split_gpuStream_t stream = 0) noexcept {
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_setInfo, sizeof(SetInfo), split_gpuCpuDeviceId, stream));
      buckets.optimizeCPU(stream);
   }

#endif

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


#ifndef HASHINATOR_CPU_ONLY_MODE
   template <typename Rule, int BLOCKSIZE = 1024>
   size_t extractPattern(KEY_TYPE* elements, Rule rule, split_gpuStream_t s = 0) {
      // Figure out Blocks to use
      size_t _s = std::ceil((float(buckets.size())) / (float)BLOCKSIZE);
      size_t nBlocks = nextPow2(_s);
      nBlocks+=(nBlocks==0);

      // Allocate with Mempool
      const size_t memory_for_pool = 8 * nBlocks * sizeof(uint32_t);
      split::tools::splitStackArena mPool(memory_for_pool, s);
      size_t retval =
          split::tools::copy_if_raw<KEY_TYPE, Rule, defaults::MAX_BLOCKSIZE, defaults::WARPSIZE>(
              buckets.data(), elements, buckets.size(), rule, nBlocks, mPool, s);
      return retval;
   }

   size_t extractAllKeys(split::SplitVector<KEY_TYPE>& elements, split_gpuStream_t s = 0, bool prefetches = true) {
      // Extract all keys
      if (prefetches){
         elements.optimizeGPU(s);
      }
      auto rule = [] __host__ __device__(const KEY_TYPE& kval) -> bool {
         return kval != EMPTYBUCKET && kval!= TOMBSTONE;
      };
      return extractPattern(elements.data(), rule, s);
   }

   void device_rehash(int newSizePower, split_gpuStream_t s = 0) {
      if (newSizePower > 32) {
         throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
      }

      size_t priorFill = _setInfo->fill;
      // Extract all valid elements
      KEY_TYPE* validElements;
      SPLIT_CHECK_ERR(split_gpuMallocAsync((void**)&validElements,
                                           (_setInfo->fill + 1) * sizeof(KEY_TYPE), s));
      optimizeGPU(s);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

      auto isValidKey = [] __host__ __device__(KEY_TYPE& element) {
         return ( (element !=TOMBSTONE) && (element!=EMPTYBUCKET) );
      };

      uint32_t nValidElements = extractPattern(validElements, isValidKey, s);

      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
      assert(nValidElements == _setInfo->fill && "Something really bad happened during rehashing! Ask Kostis!");
      // We can now clear our buckets
      // Easy optimization: If our bucket had no valid elements and the same size was requested
      // we can just clear it
      if (newSizePower == _setInfo->sizePower && nValidElements == 0) {
         clear(targets::device, s, true);
         set_status((priorFill == _setInfo->fill) ? status::success : status::fail);
         split_gpuFreeAsync(validElements, s);
         return;
      }
      optimizeCPU(s);
      buckets = std::move(split::SplitVector<KEY_TYPE>(1 << newSizePower, KEY_TYPE(EMPTYBUCKET)));
      optimizeGPU(s);
      *_setInfo = SetInfo(newSizePower);
      // Insert valid elements to now larger buckets
      insert(validElements, nValidElements, 1, s);
      set_status((priorFill == _setInfo->fill) ? status::success : status::fail);
      split_gpuFreeAsync(validElements, s);
      return;
   }


   void clean_tombstones(split_gpuStream_t s = 0, bool prefetches = false) {

      if (_setInfo->tombstoneCounter == 0) {
         return;
      }

      // Reset the tomstone counter
      _setInfo->tombstoneCounter = 0;
      // Allocate memory for overflown elements. So far this is the same size as our buckets but we can be better than
      // this

      KEY_TYPE* overflownElements;
      SPLIT_CHECK_ERR(split_gpuMallocAsync((void**)&overflownElements,
                                           (1 << _setInfo->sizePower) * sizeof(KEY_TYPE), s));

      if (prefetches) {
         optimizeGPU(s);
      }
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

      int currentSizePower = _setInfo->sizePower;
      KEY_TYPE* bck_ptr = buckets.data();

      auto isOverflown = [bck_ptr, currentSizePower] __host__ __device__(KEY_TYPE & element)->bool {
         if (element == TOMBSTONE) {
            element = EMPTYBUCKET;
            return false;
         }
         if (element == EMPTYBUCKET) {
            return false;
         }
         const size_t hashIndex = HashFunction::_hash(element, currentSizePower);
         const int bitMask = (1 << (currentSizePower)) - 1;
         bool isOverflown = (bck_ptr[hashIndex & bitMask] != element);
         return isOverflown;
      };

      // Extract overflown elements and reset overflow
      uint32_t nOverflownElements = extractPattern(overflownElements, isOverflown, s);
      _setInfo->currentMaxBucketOverflow = defaults::BUCKET_OVERFLOW;

      if (nOverflownElements == 0) {
         SPLIT_CHECK_ERR(split_gpuFreeAsync(overflownElements, s));
         return;
      }
      // If we do have overflown elements we put them back in the buckets
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
      DeviceHasher::reset_set(overflownElements, buckets.data(), _setInfo->sizePower, _setInfo->currentMaxBucketOverflow,
                          nOverflownElements, s);
      _setInfo->fill -= nOverflownElements;
      DeviceHasher::insert_set(overflownElements, buckets.data(), _setInfo->sizePower, _setInfo->currentMaxBucketOverflow,
                           &_setInfo->currentMaxBucketOverflow, &_setInfo->fill, nOverflownElements, &_setInfo->err, s);

      SPLIT_CHECK_ERR(split_gpuFreeAsync(overflownElements, s));
      return;
   }
#else
   void clean_tombstones() {
      rehash();
   }
#endif

#ifdef HASHINATOR_CPU_ONLY_MODE
   // Try to get the overflow back to the original one
   void performCleanupTasks() {
      while (_setInfo->currentMaxBucketOverflow > Hashinator::defaults::BUCKET_OVERFLOW) {
         rehash(_setInfo->sizePower + 1);
      }
      // When operating in CPU only mode we rehash to get rid of tombstones
      if (tombstone_ratio() > 0.025) {
         rehash(_setInfo->sizePower);
      }
   }
#else
   // Try to get the overflow back to the original one
   void performCleanupTasks(split_gpuStream_t s = 0) {
      while (_setInfo->currentMaxBucketOverflow > Hashinator::defaults::BUCKET_OVERFLOW) {
         device_rehash(_setInfo->sizePower + 1, s);
      }
      if (tombstone_ratio() > 0.025) {
         clean_tombstones(s);
      }
   }

#endif

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


#ifndef HASHINATOR_CPU_ONLY_MODE
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
   const_device_iterator device_begin() const noexcept {
      for (size_t i = 0; i < buckets.size(); i++) {
         if (buckets[i] != EMPTYBUCKET && buckets[i] != TOMBSTONE) {
            return const_device_iterator(*this, i);
         }
      }
      return device_end();
   }

   // Element access by iterator
   HASHINATOR_DEVICEONLY
   device_iterator device_find(KEY_TYPE key) {
      int bitMask = (1 << _setInfo->sizePower) - 1; // For efficient modulo of the array size
      auto hashIndex = hash(key);

      // Try to find the matching bucket.
      for (size_t i = 0; i < _setInfo->currentMaxBucketOverflow; i++) {
         const KEY_TYPE& candidate = buckets[(hashIndex + i) & bitMask];

         if (candidate == TOMBSTONE) {
            continue;
         }

         if (candidate == key) {
            // Found a match, return that
            return device_iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate.first == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return device_end();
         }
      }
      // Not found
      return device_end();
   }

   HASHINATOR_DEVICEONLY
   const const_device_iterator device_find(KEY_TYPE key) const {
      int bitMask = (1 << _setInfo->sizePower) - 1; // For efficient modulo of the array size
      auto hashIndex = hash(key);

      // Try to find the matching bucket.
      for (size_t i = 0; i < _setInfo->currentMaxBucketOverflow; i++) {
         const KEY_TYPE& candidate = buckets[(hashIndex + i) & bitMask];

         if (candidate == TOMBSTONE) {
            continue;
         }

         if (candidate == key) {
            // Found a match, return that
            return const_device_iterator(*this, (hashIndex + i) & bitMask);
         }

         if (candidate == EMPTYBUCKET) {
            // Found an empty bucket. Return empty.
            return device_end();
         }
      }
      // Not found
      return device_end();
   }

   HASHINATOR_DEVICEONLY
   void insert_element(const KEY_TYPE& key, size_t& thread_overflowLookup) {
      int bitMask = (1 << _setInfo->sizePower) - 1; // For efficient modulo of the array size
      auto hashIndex = hash(key);
      size_t i = 0;
      while (i < buckets.size()) {
         uint32_t vecindex = (hashIndex + i) & bitMask;
         KEY_TYPE old = split::s_atomicCAS(&buckets[vecindex], EMPTYBUCKET, key);
         // Key does not exist so we create it and incerement fill
         if (old == EMPTYBUCKET) {
            split::s_atomicAdd((unsigned int*)(&_setInfo->fill), 1);
            thread_overflowLookup = i + 1;
            return;
         }
         // Key exists so we overwrite it. Fill stays the same
         if (old == key) {
            thread_overflowLookup = i + 1;
            return;
         }
         i++;
      }
      assert(false && "Hashmap completely overflown");
   }

   HASHINATOR_DEVICEONLY
   hash_pair<device_iterator, bool> device_insert(KEY_TYPE newEntry) {
      bool found = device_find(newEntry) != device_end();
      if (!found) {
         add_element(newEntry);
      }
      return hash_pair<device_iterator, bool>(device_find(newEntry.first), !found);
   }

   HASHINATOR_DEVICEONLY
   void add_element(const KEY_TYPE& key) {
      size_t thread_overflowLookup = 0;
      insert_element(key, thread_overflowLookup);
      atomicMax((unsigned long long*)&(_setInfo->currentMaxBucketOverflow),
                nextOverflow(thread_overflowLookup, defaults::WARPSIZE / defaults::elementsPerWarp));
   }
#endif

   void print_pair(const KEY_TYPE& i) const noexcept {
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

   void dump_buckets() const noexcept {
      printf("Hashinator Stats \n");
      printf("Fill= %zu, LoadFactor=%f \n", _setInfo->fill, load_factor());
      printf("Tombstones= %zu\n", _setInfo->tombstoneCounter);
      for (size_t i = 0; i < buckets.size(); ++i) {
         print_pair(buckets[i]);
      }
      printf("\n");
   }

   HASHINATOR_HOSTDEVICE
   void stats() const noexcept{
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

   HASHINATOR_HOSTDEVICE
   float tombstone_ratio() const noexcept {
      if (tombstone_count() == 0) {
         return 0.0;
      }
      return (float)_setInfo->tombstoneCounter / (float)buckets.size();
   }

   bool contains(const KEY_TYPE& key) const noexcept { return (find(key) != end()) ; }

   bool empty() const noexcept { return begin() == end(); }

   size_t count(const KEY_TYPE& key) const noexcept { return contains(key) ? 1 : 0; }

#ifdef HASHINATOR_CPU_ONLY_MODE
   void clear(targets t= targets::host){
      UNUSED(t);
      buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, {EMPTYBUCKET});
      *_setInfo = SetInfo(_setInfo->sizePower);
      return;
   }
#else
   void clear(targets t = targets::host, split_gpuStream_t s = 0, bool prefetches = true) {
      switch (t) {
      case targets::host:
         buckets = split::SplitVector<KEY_TYPE>(1 << _setInfo->sizePower, {EMPTYBUCKET});
         *_setInfo = SetInfo(_setInfo->sizePower);
         break;
      case targets::device:
         if (prefetches) {
            buckets.optimizeGPU(s);
         }
         DeviceHasher::reset_all_set(buckets.data(), buckets.size(), s);
         _setInfo->fill = 0;
         set_status((_setInfo->fill == 0) ? success : fail);
         break;
      default:
         clear(targets::host);
         break;
      }
      return;
   }
#endif


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
      performCleanupTasks();
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
      assert(index <static_cast<size_t>(  1 << _setInfo->sizePower ));
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

   bool erase(const KEY_TYPE& key) {
      auto it = find(key);
      if (it!=end()){
         erase(it);
         return true;
      }
      return false;
   }


#ifdef HASHINATOR_CPU_ONLY_MODE
   void resize(int newSizePower,targets t = targets::host) {
      UNUSED(t);
      rehash(newSizePower);
   }

   void insert(KEY_TYPE* keys,size_t len,float targetLF = 0.5) {
      UNUSED(targetLF);
      for (size_t i =0 ; i < len; ++i){
         insert(keys[i]);

      }
   }

   void erase(KEY_TYPE* keys,size_t len,float targetLF = 0.5) {
      UNUSED(targetLF);
      for (size_t i =0 ; i < len; ++i){
         erase(keys[i]);

      }
   }
#else
   void resize(int newSizePower, targets t = targets::host, split_gpuStream_t s = 0) {
      switch (t) {
      case targets::host:
         rehash(newSizePower);
         break;
      case targets::device:
         device_rehash(newSizePower, s);
         break;
      default:
         resize(newSizePower, targets::host);
         break;
      }
      return;
   }
   

   void insert(KEY_TYPE* keys,size_t len,float targetLF = 0.5, split_gpuStream_t s = 0, bool prefetches = true) {
      // TODO fix these if paths or at least annotate them .
      if (len == 0) {
         set_status(status::success);
         return;
      }
      if (prefetches) {
         buckets.optimizeGPU(s);
      }
      int64_t neededPowerSize = std::ceil(std::log2((_setInfo->fill + len) * (1.0 / targetLF)));
      if (neededPowerSize > _setInfo->sizePower) {
         resize(neededPowerSize, targets::device, s);
      }
      _setInfo->currentMaxBucketOverflow = _setInfo->currentMaxBucketOverflow;
      DeviceHasher::insert_set(keys, buckets.data(), _setInfo->sizePower, _setInfo->currentMaxBucketOverflow,
                           &_setInfo->currentMaxBucketOverflow, &_setInfo->fill, len, &_setInfo->err, s);
      return;
   }

   // Uses Hasher's erase_kernel to delete  elements
   void erase(KEY_TYPE* keys, size_t len, split_gpuStream_t s = 0) {
      if (len == 0) {
         set_status(status::success);
         return;
      }
      buckets.optimizeGPU(s);
      // Remember the last number of tombstones
      size_t tbStore = tombstone_count();
      DeviceHasher::erase_set(keys, buckets.data(), &_setInfo->tombstoneCounter, _setInfo->sizePower,
                          _setInfo->currentMaxBucketOverflow, len, s);
      size_t tombstonesAdded = tombstone_count() - tbStore;
      // Fill should be decremented by the number of tombstones added;
      _setInfo->fill -= tombstonesAdded;
      return;
   }

#endif
   
#ifndef HASHINATOR_CPU_ONLY_MODE
   template <bool skipOverWrites = false>
   HASHINATOR_DEVICEONLY void warpInsert(const KEY_TYPE& candidateKey, const size_t w_tid) noexcept {

      const int sizePower = _setInfo->sizePower;
      const int bitMask = (1 << (sizePower)) - 1;
      const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
      const size_t optimalindex = (hashIndex)&bitMask;
      const auto submask = SPLIT_VOTING_MASK;
      bool warpDone = false;
      uint64_t threadOverflow = 1;

#ifdef HASHINATOR_DEBUG
// Safety check: make sure everyone has the same key/val and all threads are here.
#ifdef __CUDACC__
      assert(__activemask() == SPLIT_VOTING_MASK && "Tried to warpInsert with part of warp predicated off");
#endif
      KEY_TYPE storeKey = split::s_shuffle(candidateKey, 0, SPLIT_VOTING_MASK);
      bool isSafe = (split::s_warpVote(candidateKey == storeKey, SPLIT_VOTING_MASK) == SPLIT_VOTING_MASK);
      assert(isSafe && "Tried to warpInsert with different keys in the same warp");
#endif

      for (size_t i = 0; i < (1 << sizePower); i += defaults::WARPSIZE) {
         // Check if this virtual warp is done.
         if (warpDone) {
            break;
         }

         // Get the position we should be looking into
         size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
         auto target = buckets[probingindex];

         // vote for available emptybuckets in warp region
         // Note that this has to be done before voting for already existing elements (below)
         auto mask = split::s_warpVote(target == EMPTYBUCKET, submask);

         // Check if this elements already exists
         auto already_exists = split::s_warpVote(target == candidateKey, submask);
         if (already_exists) {
            int winner = split::s_findFirstSig(already_exists) - 1;
            if (w_tid == winner) {
               warpDone = 1;
            }
         }

         // If any duplicate was there now is the time for the whole Virtual warp to find out!
         warpDone = split::s_warpVote(warpDone > 0, submask) & submask;

         while (mask && !warpDone) {
            int winner = split::s_findFirstSig(mask) - 1;
            if (w_tid == winner) {
               KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex], EMPTYBUCKET, candidateKey);
               if (old == EMPTYBUCKET) {
                  threadOverflow = (probingindex < optimalindex) ? (1 << sizePower) : (probingindex - optimalindex + 1);
                  warpDone = 1;
                  split::s_atomicAdd(&_setInfo->fill, 1);
                  if (threadOverflow > _setInfo->currentMaxBucketOverflow) {
                     split::s_atomicExch((unsigned long long*)(&_setInfo->currentMaxBucketOverflow),
                                         (unsigned long long)nextOverflow(threadOverflow, defaults::WARPSIZE));
                  }
               } else if (old == candidateKey) {
                  warpDone = 1;
               }
            }
            // If any of the virtual warp threads are done the the whole
            // Virtual warp is done
            warpDone = split::s_warpVote(warpDone > 0, submask);
            mask ^= (1UL << winner);
         }
      }
   }

   template <bool skipOverWrites = false>
   HASHINATOR_DEVICEONLY bool warpInsert_V(const KEY_TYPE& candidateKey,const size_t w_tid) noexcept {

      const int sizePower = _setInfo->sizePower;
      const int bitMask = (1 << (sizePower)) - 1;
      const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
      const size_t optimalindex = (hashIndex)&bitMask;
      const auto submask = SPLIT_VOTING_MASK;
      bool warpDone = false;
      uint64_t threadOverflow = 1;
      int localCount = 0;

#ifdef HASHINATOR_DEBUG
// Safety check: make sure everyone has the same key/val and all threads are here.
#ifdef __CUDACC__
      assert(__activemask() == SPLIT_VOTING_MASK && "Tried to warpInsert_V with part of warp predicated off");
#endif
      KEY_TYPE storeKey = split::s_shuffle(candidateKey, 0, SPLIT_VOTING_MASK);
      KEY_TYPE storeVal = split::s_shuffle(candidateVal, 0, SPLIT_VOTING_MASK);
      bool isSafe = (split::s_warpVote(candidateKey == storeKey, SPLIT_VOTING_MASDK) == SPLIT_VOTING_MASK;
      assert(isSafe && "Tried to warpInsert_V with different keys in the same warp");
#endif

      for (size_t i = 0; i < (1 << sizePower); i += defaults::WARPSIZE) {
         // Check if this virtual warp is done.
         if (warpDone) {
            break;
         }

         // Get the position we should be looking into
         size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
         auto target = buckets[probingindex];

         // vote for available emptybuckets in warp region
         // Note that this has to be done before voting for already existing elements (below)
         auto mask = split::s_warpVote(target == EMPTYBUCKET, submask);

         // Check if this elements already exists
         auto already_exists = split::s_warpVote(target == candidateKey, submask);
         if (already_exists) {
            int winner = split::s_findFirstSig(already_exists) - 1;
            if (w_tid == winner) {
               warpDone = 1;
            }
         }

         // If any duplicate was there now is the time for the whole Virtual warp to find out!
         warpDone = split::s_warpVote(warpDone > 0, submask) & submask;

         while (mask && !warpDone) {
            int winner = split::s_findFirstSig(mask) - 1;
            if (w_tid == winner) {
               KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex], EMPTYBUCKET, candidateKey);
               if (old == EMPTYBUCKET) {
                  threadOverflow = (probingindex < optimalindex) ? (1 << sizePower) : (probingindex - optimalindex + 1);
                  warpDone = 1;
                  localCount = 1;
                  split::s_atomicAdd(&_setInfo->fill, 1);
                  if (threadOverflow > _setInfo->currentMaxBucketOverflow) {
                     split::s_atomicExch((unsigned long long*)(&_setInfo->currentMaxBucketOverflow),
                                         (unsigned long long)nextOverflow(threadOverflow, defaults::WARPSIZE));
                  }
               } else if (old == candidateKey) {
                  warpDone = 1;
               }
            }
            // If any of the virtual warp threads are done the the whole
            // Virtual warp is done
            warpDone = split::s_warpVote(warpDone > 0, submask);
            mask ^= (1UL << winner);
         }
      }

      auto res = split::s_warpVote(localCount > 0, submask);
      return (res > 0);
   }

   HASHINATOR_DEVICEONLY
   void warpErase(const KEY_TYPE& candidateKey, const size_t w_tid) noexcept {

      const int sizePower = _setInfo->sizePower;
      const size_t maxoverflow = _setInfo->currentMaxBucketOverflow;
      const int bitMask = (1 << (sizePower)) - 1;
      const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
      const auto submask = SPLIT_VOTING_MASK;
      bool warpDone = false;
      int winner = 0;

#ifdef HASHINATOR_DEBUG
// Safety check: make sure everyone has the same key/val and all threads are here.
#ifdef __CUDACC__
      assert(__activemask() == SPLIT_VOTING_MASK && "Tried to warpFind with part of warp predicated off");
#endif
      KEY_TYPE storeKey = split::s_shuffle(candidateKey, 0, SPLIT_VOTING_MASK);
      bool isSafe = split::s_warpVote(candidateKey == storeKey, SPLIT_VOTING_MASK) == SPLIT_VOTING_MASK;
      assert(isSafe && "Tried to warpFind with different keys/vals in the same warp");
#endif

      for (size_t i = 0; i < maxoverflow; i += defaults::WARPSIZE) {

         if (warpDone) {
            break;
         }

         // Get the position we should be looking into
         size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
         const auto maskExists =
             split::s_warpVote(buckets[probingindex] == candidateKey, SPLIT_VOTING_MASK) & submask;
         const auto emptyFound =
             split::s_warpVote(buckets[probingindex] == EMPTYBUCKET, SPLIT_VOTING_MASK) & submask;
         // If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
         if (!maskExists && emptyFound) {
            warpDone = true;
         }
         if (maskExists) {
            winner = split::s_findFirstSig(maskExists) - 1;
            if (w_tid == winner) {
               buckets[probingindex] = TOMBSTONE;
               split::s_atomicAdd(&_setInfo->tombstoneCounter, 1);
               split::s_atomicSub((unsigned int*)&_setInfo->fill, 1);
            }
            warpDone = true;
         }
      }
      return;
   }
#endif

}; // Unordered_Set

} // namespace Hashinator

/* File:    hashinator.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
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
#include "../common.h"
#include "../splitvector/splitvec.h"
#include "../splitvector/split_allocators.h"
#include "hashfunctions.h"
#include "defaults.h"
#include "hash_pair.h"
#ifndef HASHINATOR_HOST_ONLY
#include "../splitvector/split_tools.h"
#include "hashers.h"
#endif

namespace Hashinator{

   #ifndef HASHINATOR_HOST_ONLY
   template <typename T>
   using DefaultMetaAllocator = split::split_unified_allocator<T>;
   #define DefaultHasher Hashers::Hasher<KEY_TYPE,VAL_TYPE,HashFunction,EMPTYBUCKET,TOMBSTONE,defaults::WARPSIZE,defaults::elementsPerWarp>
   #else
   template <typename T>
   using DefaultMetaAllocator = split::split_host_allocator<T>;
   #define DefaultHasher void
   #endif

   typedef struct Info {
      Info(int sz)
         :sizePower(sz),fill(0),currentMaxBucketOverflow(defaults::BUCKET_OVERFLOW),tombstoneCounter(0),err(status::invalid){}
      int sizePower;
      size_t fill; 
      size_t currentMaxBucketOverflow;
      size_t tombstoneCounter;
      status err;
   }MapInfo;

   template <typename KEY_TYPE, 
             typename VAL_TYPE, 
             KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
             KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1,
             class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
             class DeviceHasher=DefaultHasher,
             class Meta_Allocator=DefaultMetaAllocator<MapInfo>>
   class Hashmap {

   private:
      
      //CUDA device handle
      Hashmap* device_map;
      //~CUDA device handle

      //Host members
      split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> buckets;
      Meta_Allocator _metaAllocator;    // Allocator used to allocate and deallocate memory for metadata
      MapInfo* _mapInfo;
      //~Host members


       // Wrapper over available hash functions 
      HASHINATOR_HOSTDEVICE
      uint32_t hash(KEY_TYPE in) const {
          static_assert(std::is_arithmetic<KEY_TYPE>::value && sizeof(KEY_TYPE) <= sizeof(uint32_t));
          return HashFunction::_hash(in,_mapInfo->sizePower);
       }
      
      // Used by the constructors. Preallocates the device pointer and bookeepping info for later use on device. 
      // This helps in reducing the number of calls to hipMalloc
      HASHINATOR_HOSTONLY
      void preallocate_device_handles(){
         #ifndef HASHINATOR_HOST_ONLY
         hipMalloc((void **)&device_map, sizeof(Hashmap));
         #endif
      }

      // Deallocates the bookeepping info and the device pointer
      HASHINATOR_HOSTONLY
      void deallocate_device_handles(){
         #ifndef HASHINATOR_HOST_ONLY
         hipFree(device_map);
         device_map=nullptr;
         #endif
      }   

      HASHINATOR_HOSTDEVICE
      inline void set_status(status code)noexcept{
         _mapInfo->err=code;
      }

   public:
      template <typename T, typename U>
      struct Overflown_Predicate{

      hash_pair<KEY_TYPE, VAL_TYPE> *bck_ptr;
      int currentSizePower;
   
      explicit Overflown_Predicate(hash_pair<KEY_TYPE, VAL_TYPE>*ptr,int s):bck_ptr(ptr),currentSizePower(s){}
      Overflown_Predicate()=delete;
         HASHINATOR_HOSTDEVICE
         inline bool operator()( hash_pair<T,U>& element)const{
            if (element.first==TOMBSTONE){element.first=EMPTYBUCKET;return false;}
            if (element.first==EMPTYBUCKET){return false;}
            const size_t hashIndex = HashFunction::_hash(element.first,currentSizePower);
            const int bitMask = (1 <<(currentSizePower )) - 1; 
            bool isOverflown=(bck_ptr[hashIndex&bitMask].first!=(int)element.first);
            return isOverflown;
         }
      };

      template <typename T, typename U>
      struct Valid_Predicate{
      Valid_Predicate(){}
         HASHINATOR_HOSTDEVICE
         inline bool operator()( hash_pair<T,U>& element)const{
            if (element.first!=TOMBSTONE && element.first!=EMPTYBUCKET  ){return true;}
            return false;
         }
      };

      HASHINATOR_HOSTONLY
      Hashmap(){
         preallocate_device_handles();
         _mapInfo=_metaAllocator.allocate(1);
         *_mapInfo=MapInfo(6);
         buckets=split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> (1 << _mapInfo->sizePower, hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
       };

      HASHINATOR_HOSTONLY
      Hashmap(int sizepower){
         preallocate_device_handles();
         _mapInfo=_metaAllocator.allocate(1);
         *_mapInfo=MapInfo(sizepower);
         buckets=split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> (1 << _mapInfo->sizePower, hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
       };

      HASHINATOR_HOSTONLY
      Hashmap(const Hashmap<KEY_TYPE, VAL_TYPE>& other){
         preallocate_device_handles();
         _mapInfo=_metaAllocator.allocate(1);
         *_mapInfo=*(other._mapInfo);
         buckets=other.buckets;
       };

      HASHINATOR_HOSTONLY
      ~Hashmap(){     
         deallocate_device_handles();
         _metaAllocator.deallocate(_mapInfo,1);
      };

      #ifdef HASHINATOR_HOST_ONLY
      HASHINATOR_HOSTONLY
      void *operator new(size_t len) {
         void *ptr = (void*)malloc(len) ;
         return ptr;
      }

      HASHINATOR_HOSTONLY
      void operator delete(void *ptr) {
         free(ptr);
      }

      HASHINATOR_HOSTONLY
      void* operator new[] (size_t len) {
         void *ptr = (void*)malloc(len);
         return ptr;
      }

      HASHINATOR_HOSTONLY
      void operator delete[] (void* ptr) {
         free(ptr);
      }

      #else
      HASHINATOR_HOSTONLY
      void *operator new(size_t len) {
         void *ptr;
         hipMallocManaged(&ptr, len);
         return ptr;
      }

      HASHINATOR_HOSTONLY
      void operator delete(void *ptr) {
         hipFree(ptr);
      }

      HASHINATOR_HOSTONLY
      void* operator new[] (size_t len) {
         void *ptr;
         hipMallocManaged(&ptr, len);
         return ptr;
      }

      HASHINATOR_HOSTONLY
      void operator delete[] (void* ptr) {
         hipFree(ptr);
      }
      #endif

      HASHINATOR_HOSTONLY
      void copyMetadata(MapInfo* dst,cudaStream_t s=0){
         cudaMemcpyAsync(dst, _mapInfo, sizeof(MapInfo),cudaMemcpyDeviceToHost,s);
      }

      // Resize the table to fit more things. This is automatically invoked once
      // maxBucketOverflow has triggered. This can only be done on host (so far)
      HASHINATOR_HOSTONLY
      void rehash(int newSizePower) {
         if (newSizePower > 32) {
            throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
         }
         split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> newBuckets(1 << newSizePower,
                                                     hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
         _mapInfo->sizePower = newSizePower;
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size

         // Iterate through all old elements and rehash them into the new array.
         for (auto& e : buckets) {
            // Skip empty buckets ; We also check for TOMBSTONE elements
            // as we might be coming off a kernel that overflew the hashmap
            if (e.first == EMPTYBUCKET || e.first==TOMBSTONE) {
               continue;
            }

            uint32_t newHash = hash(e.first);
            bool found = false;
            for (int i = 0; i < Hashinator::defaults::BUCKET_OVERFLOW; i++) {
               hash_pair<KEY_TYPE, VAL_TYPE>& candidate = newBuckets[(newHash + i) & bitMask];
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
         buckets = newBuckets;
         _mapInfo->currentMaxBucketOverflow=Hashinator::defaults::BUCKET_OVERFLOW;
         _mapInfo->tombstoneCounter=0;
      }


      #ifndef HASHINATOR_HOST_ONLY
      // Resize the table to fit more things. This is automatically invoked once
      // maxBucketOverflow has triggered. This can only be done on host (so far)
      HASHINATOR_HOSTONLY
      void device_rehash(int newSizePower, hipStream_t s=0) {
         if (newSizePower > 32) {
            throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
         }

         size_t priorFill=_mapInfo->fill;
         //Extract all valid elements
         hash_pair<KEY_TYPE, VAL_TYPE>* validElements; 
         hipMallocAsync((void**)&validElements, (_mapInfo->fill+1) * sizeof(hash_pair<KEY_TYPE, VAL_TYPE>),s);
         optimizeGPU(s);
         hipStreamSynchronize(s);


         uint32_t nValidElements = split::tools::copy_if_raw
                  <hash_pair<KEY_TYPE, VAL_TYPE>,Valid_Predicate<KEY_TYPE,VAL_TYPE>,defaults::MAX_BLOCKSIZE,defaults::WARPSIZE>
                  (buckets,validElements,Valid_Predicate<KEY_TYPE,VAL_TYPE>(),s);


         hipStreamSynchronize(s);
         assert(nValidElements==_mapInfo->fill && "Something really bad happened during rehashing! Ask Kostis!");
         //We can now clear our buckets
         optimizeCPU(s);
         buckets=std::move(split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>> (1 << newSizePower, hash_pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE())));
         optimizeGPU(s);
         *_mapInfo=Info(newSizePower);
         //Insert valid elements to now larger buckets
         insert(validElements,nValidElements,1,s);
         set_status((priorFill==_mapInfo->fill)?status::success:status::fail);
      }
      #endif


      // Element access (by reference). Nonexistent elements get created.
      HASHINATOR_HOSTONLY
      VAL_TYPE& _at(const KEY_TYPE& key) {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            
            hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
            
            if (candidate.first == key) {
               // Found a match, return that
               return candidate.second;
            }

            if (candidate.first == EMPTYBUCKET) {
               // Found an empty bucket, assign and return that.
               candidate.first = key;
               _mapInfo->fill++;
               return candidate.second;
            }

            if (candidate.first == TOMBSTONE) {
               bool alreadyExists=false;

               //We remove this Tombstone 
               candidate.first = key;
               _mapInfo->tombstoneCounter--;

               //We look ahead in case candidate was already in the hashmap
               //If we find it then we swap the duplicate with empty and do not increment fill
               //but we only reduce the tombstone count
               for (size_t j = i+1; j < _mapInfo->currentMaxBucketOverflow; ++j) {
                  hash_pair<KEY_TYPE, VAL_TYPE>& duplicate = buckets[(hashIndex + j) & bitMask];
                  if (duplicate.first==candidate.first){
                     alreadyExists=true;
                     candidate.second=duplicate.second;
                     if ( buckets[(hashIndex + j+1) & bitMask].first==EMPTYBUCKET || j+1>=_mapInfo->currentMaxBucketOverflow){
                        duplicate.first=EMPTYBUCKET;
                     }else{
                        duplicate.first=TOMBSTONE;
                        _mapInfo->tombstoneCounter++;
                     }
                     break;
                  }
               }
               if (!alreadyExists){
                  _mapInfo->fill++;
               }
               return candidate.second;
            }
         }

         // Not found, and we have no free slots to create a new one. So we need to rehash to a larger size.
         #ifdef HASHINATOR_HOST_ONLY
         rehash(_mapInfo->sizePower + 1);
         #else
         device_rehash(_mapInfo->sizePower + 1);
         assert(peek_status()==status::success);
         #endif
         return at(key); // Recursive tail call to try again with larger table.
      }

      HASHINATOR_HOSTONLY
      const VAL_TYPE& _at(const KEY_TYPE& key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

            if (candidate.first == TOMBSTONE) {
               continue;
            }

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

      HASHINATOR_HOSTDEVICE
      inline status peek_status(void) noexcept{
         status retval = _mapInfo->err;
         _mapInfo->err=status::invalid;
         return retval;
      }

      HASHINATOR_HOSTDEVICE
      inline int getSizePower(void) const noexcept{
         return _mapInfo->sizePower;
      }

      // For STL compatibility: size(), bucket_count(), count(KEY_TYPE), clear()
      HASHINATOR_HOSTDEVICE
      size_t size() const { return _mapInfo->fill; }

      HASHINATOR_HOSTDEVICE
      size_t bucket_count() const {
         return buckets.size();
      }
      
      HASHINATOR_HOSTONLY
      float load_factor() const {return (float)size()/bucket_count();}

      HASHINATOR_HOSTONLY
      size_t count(const KEY_TYPE& key) const {
         if (find(key) != end()) {
            return 1;
         } else {
            return 0;
         }
      }

      #ifdef HASHINATOR_HOST_ONLY
      HASHINATOR_HOSTONLY
      void clear() {
         buckets = split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>>(1 << _mapInfo->sizePower, {EMPTYBUCKET, VAL_TYPE()});
         *_mapInfo=MapInfo(_mapInfo->sizePower);
         return;
      }
      #else
      HASHINATOR_HOSTONLY
      void clear(targets t=targets::host, hipStream_t s=0) {
         size_t blocksNeeded;
         switch(t)
         {
            case targets::host :
               buckets = split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>>(1 << _mapInfo->sizePower, {EMPTYBUCKET, VAL_TYPE()});
               *_mapInfo=MapInfo(_mapInfo->sizePower);
               break;

            case targets::device :
               buckets.optimizeGPU(s);
               blocksNeeded=buckets.size()/defaults::MAX_BLOCKSIZE;
               blocksNeeded=blocksNeeded+(blocksNeeded==0);
               Hashers::reset_all_to_empty<KEY_TYPE,VAL_TYPE,EMPTYBUCKET>
               <<<blocksNeeded,defaults::MAX_BLOCKSIZE,0,s>>>(buckets.data(),buckets.size(),&_mapInfo->fill);
               hipStreamSynchronize(s);
               set_status(( _mapInfo->fill==0 )?success:fail);
               break;

            default:
               clear(targets::host);
               break;
         }
         return;
      }
      #endif


      //Try to grow our buckets until we achieve a targetLF load factor
      HASHINATOR_HOSTONLY
      void resize_to_lf(float targetLF=0.5){
         while (load_factor() > targetLF){
            rehash(_mapInfo->sizePower+1);
         }
      }

      #ifdef HASHINATOR_HOST_ONLY
      HASHINATOR_HOSTONLY
      void resize(int newSizePower){
         rehash(newSizePower);     
      }
      #else
      HASHINATOR_HOSTONLY
      void resize(int newSizePower,targets t=targets::host,hipStream_t s=0){
         switch(t)
         {
            case targets::host:
               rehash(newSizePower);     
               break;
            case targets::device:
               device_rehash(newSizePower,s);
               break;
            default:
               std::cerr<<"Defaulting to host rehashing"<<std::endl;
               resize(newSizePower,targets::host);
               break;
         }
         return;
      }
      #endif

      HASHINATOR_HOSTONLY
         void print_pair(const hash_pair<KEY_TYPE, VAL_TYPE>& i)const {
            size_t currentSizePower=_mapInfo->sizePower;
            const size_t hashIndex = HashFunction::_hash(i.first,currentSizePower);
            const int bitMask = (1 <<(currentSizePower )) - 1; 
            size_t optimalIndex=hashIndex&bitMask;
            const_iterator it=find(i.first);
            int64_t overflow=llabs(it.getIndex()-optimalIndex);
            if (i.first==TOMBSTONE){
               std::cout<<"[╀] ";
            }else if (i.first == EMPTYBUCKET){
               std::cout<<"[▢] ";
            }
            else{
               if (overflow>0){
                  printf("[%d,%d,\033[1;31m%li\033[0m] ",i.first,i.second,overflow);
               }else{
                  printf("[%d,%d,%zu] ",i.first,i.second,overflow);
               }
            }
         }

      HASHINATOR_HOSTONLY
      void dump_buckets()const {
         printf("Hashinator Stats \n");
         printf("Fill= %zu, LoadFactor=%f \n",_mapInfo->fill,load_factor());
         printf("Tombstones= %zu\n",_mapInfo->tombstoneCounter);
         for  (int i =0 ; i < buckets.size(); ++i){
            print_pair(buckets[i]);
         }
         std::cout<<std::endl;

      }

      HASHINATOR_HOSTONLY
      void stats()const {
         printf("Hashinator Stats \n");
         printf("Bucket size= %zu\n",buckets.size());
         printf("Fill= %zu, LoadFactor=%f \n",_mapInfo->fill,load_factor());
         printf("Tombstones= %zu\n",_mapInfo->tombstoneCounter);
         printf("Overflow= %zu\n",_mapInfo->currentMaxBucketOverflow);

      }

      HASHINATOR_HOSTONLY
      size_t tombstone_count()const {
         return _mapInfo->tombstoneCounter;
      }

      HASHINATOR_HOSTONLY
      float tombstone_ratio()const {
         if (tombstone_count()==0){
            return 0.0;
         }

         return (float)_mapInfo->tombstoneCounter/(float)buckets.size();
      }

      HASHINATOR_HOSTONLY
      void swap(Hashmap<KEY_TYPE, VAL_TYPE>& other) noexcept{
         buckets.swap(other.buckets);
         std::swap(_mapInfo,other._mapInfo);
         std::swap(device_map,other.device_map);
      }


      #ifdef HASHINATOR_HOST_ONLY
      //Try to get the overflow back to the original one
      HASHINATOR_HOSTONLY
      void performCleanupTasks(){
         while (_mapInfo->currentMaxBucketOverflow > Hashinator::defaults::BUCKET_OVERFLOW){
            rehash(_mapInfo->sizePower+1);
         }
         //When operating in CPU only mode we rehash to get rid of tombstones
         if (tombstone_ratio()>0.25){
            rehash(_mapInfo->sizePower);
         }
      }
      #else
      //Try to get the overflow back to the original one 
      HASHINATOR_HOSTONLY
      void performCleanupTasks(hipStream_t s=0){
         while (_mapInfo->currentMaxBucketOverflow > Hashinator::defaults::BUCKET_OVERFLOW){
            rehash(_mapInfo->sizePower+1);
         }
         if (tombstone_ratio()>0.25){
            clean_tombstones(s);
         }
      }

      #endif

      //Read only  access to reference. 
      HASHINATOR_HOSTONLY
      const VAL_TYPE& at(const KEY_TYPE& key) const {
         performCleanupTasks();
         return _at(key);
      }

      //See _at(key)
      HASHINATOR_HOSTONLY
      VAL_TYPE& at(const KEY_TYPE& key) {
         performCleanupTasks();
         return _at(key);
      }

      // Typical array-like access with [] operator
      HASHINATOR_HOSTONLY
      VAL_TYPE& operator[](const KEY_TYPE& key) {
         performCleanupTasks();
         return at(key); 
      }
      
      
      // Iterator type. Iterates through all non-empty buckets.
      class iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<KEY_TYPE, VAL_TYPE>> {
         Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
         size_t index;

      public:
         HASHINATOR_HOSTONLY
         iterator(Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

         HASHINATOR_HOSTONLY
         iterator& operator++() {
            index++;
            while(index < hashtable->buckets.size()){
               if (hashtable->buckets[index].first != EMPTYBUCKET && hashtable->buckets[index].first != TOMBSTONE  ){
                  break;
               }
               index++;
            }
            return *this;
         }
         
         HASHINATOR_HOSTONLY
         iterator operator++(int) { // Postfix version
            iterator temp = *this;
            ++(*this);
            return temp;
         }
         HASHINATOR_HOSTONLY
         bool operator==(iterator other) const {
            return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
         }
         HASHINATOR_HOSTONLY
         bool operator!=(iterator other) const {
            return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
         }
         HASHINATOR_HOSTONLY
         hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         size_t getIndex() { return index; }
      };

      // Const iterator.
      class const_iterator : public std::iterator<std::random_access_iterator_tag, hash_pair<KEY_TYPE, VAL_TYPE>> {
         const Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
         size_t index;

      public:
         HASHINATOR_HOSTONLY
         explicit const_iterator(const Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index)
             : hashtable(&hashtable), index(index) {}
         HASHINATOR_HOSTONLY
         const_iterator& operator++() {
            index++;
            while(index < hashtable->buckets.size()){
               if (hashtable->buckets[index].first != EMPTYBUCKET && hashtable->buckets[index].first != TOMBSTONE){
                  break;
               }
               index++;
            }
            return *this;
         }
         HASHINATOR_HOSTONLY
         const_iterator operator++(int) { // Postfix version
            const_iterator temp = *this;
            ++(*this);
            return temp;
         }
         HASHINATOR_HOSTONLY
         bool operator==(const_iterator other) const {
            return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
         }
         HASHINATOR_HOSTONLY
         bool operator!=(const_iterator other) const {
            return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
         }
         HASHINATOR_HOSTONLY
         const hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         const hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         size_t getIndex() { return index; }
      };

      // Element access by iterator
      HASHINATOR_HOSTONLY
      const const_iterator find(KEY_TYPE key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
            
            if(candidate.first==TOMBSTONE){continue;}

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

      HASHINATOR_HOSTONLY
      iterator find(KEY_TYPE key) {
         performCleanupTasks();
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

            if(candidate.first==TOMBSTONE){continue;}
            
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
      
      HASHINATOR_HOSTONLY
      iterator begin() {
         for (size_t i = 0; i < buckets.size(); i++) {
            if (buckets[i].first != EMPTYBUCKET && buckets[i].first!=TOMBSTONE) {
               return iterator(*this, i);
            }
         }
         return end();
      }

      HASHINATOR_HOSTONLY
      const_iterator begin() const {
         for (size_t i = 0; i < buckets.size(); i++) {
            if (buckets[i].first != EMPTYBUCKET && buckets[i].first!=TOMBSTONE) {
               return const_iterator(*this, i);
            }
         }
         return end();
      }

      HASHINATOR_HOSTONLY
      iterator end() { return iterator(*this, buckets.size()); }

      HASHINATOR_HOSTONLY
      const_iterator end() const { return const_iterator(*this, buckets.size()); }

      // Remove one element from the hash table.
      HASHINATOR_HOSTONLY
      iterator erase(iterator keyPos) {
         size_t index = keyPos.getIndex();
         if (buckets[index].first != EMPTYBUCKET && buckets[index].first !=TOMBSTONE ){
            buckets[index].first=TOMBSTONE;
            _mapInfo->fill--;
            _mapInfo->tombstoneCounter++;
         }
         // return the next valid bucket member
         ++keyPos;
         return keyPos;
      }

      HASHINATOR_HOSTONLY
      hash_pair<iterator, bool> insert(hash_pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = find(newEntry.first) != end();
         if (!found) {
            at(newEntry.first) = newEntry.second;
         }
         return hash_pair<iterator, bool>(find(newEntry.first), !found);
      }

      HASHINATOR_HOSTONLY
      size_t erase(const KEY_TYPE& key) {
         iterator element = find(key);
         if(element == end()) {
            return 0;
         } else {
            erase(element);
            return 1;
         }
      }
      

      #ifndef HASHINATOR_HOST_ONLY

      /*
       * Fills the splitvector "elements" with **copies** of the keys that match the pattern
       * dictated by Rule. 
       * Example Usage:
       *
       * Define this somewhere: 
       *
       *  template <typename T, typename U>
       *  struct Rule{
       *  Rule(){}
       *     __host__ __device__
       *     inline bool operator()( hash_pair<T,U>& element)const{
       *        if (element.first<100 ){return true;}
       *        return false;
       *     }
       *  };
       *
       * Then call this:
       *   hmap.extractPattern(elements,Rule<uint32_t,uint32_t>());
       * */
      template <typename  Rule>
      size_t extractPattern(split::SplitVector<hash_pair<KEY_TYPE, VAL_TYPE>>& elements ,Rule, hipStream_t s=0){
         elements.resize(1<<_mapInfo->sizePower);
         elements.optimizeGPU(s);
         //Extract elements matching the Pattern Rule(element)==true;
         size_t retval = split::tools::copy_if_raw<hash_pair
                         <KEY_TYPE, VAL_TYPE>,Rule,defaults::MAX_BLOCKSIZE,defaults::WARPSIZE>
                         (buckets,elements.data(),Rule(),s);

         //Remove unwanted elements
         elements.erase(&elements->at(retval),elements->end());
         return retval;
      }

      template <typename  Rule>
      size_t extractKeysByPattern(split::SplitVector<KEY_TYPE>& elements ,Rule, hipStream_t s=0){
         elements.resize(_mapInfo->fill);
         elements.optimizeGPU(s);
         //Extract element **keys** matching the Pattern Rule(element)==true;
         size_t retval=split::tools::copy_keys_if_raw
                       <hash_pair<KEY_TYPE, VAL_TYPE>,KEY_TYPE,Rule,defaults::MAX_BLOCKSIZE,defaults::WARPSIZE>
                       (buckets,elements.data(),Rule(),s);
         return retval;
      }

      void clean_tombstones(hipStream_t s=0){
   
         if (_mapInfo->tombstoneCounter == 0){
            return ;
         }

         //Reset the tomstone counter
         _mapInfo->tombstoneCounter=0;
         //Allocate memory for overflown elements. So far this is the same size as our buckets but we can be better than this 
         
         hash_pair<KEY_TYPE, VAL_TYPE>* overflownElements; 
         hipMallocAsync((void**)&overflownElements, (1<<_mapInfo->sizePower) * sizeof(hash_pair<KEY_TYPE, VAL_TYPE>),s);

         optimizeGPU(s);
         hipStreamSynchronize(s);
         uint32_t nOverflownElements=split::tools::copy_if_raw
            <hash_pair<KEY_TYPE, VAL_TYPE>,Overflown_Predicate<KEY_TYPE,VAL_TYPE>,defaults::MAX_BLOCKSIZE,defaults::WARPSIZE>
            (buckets,
             overflownElements,
             Overflown_Predicate<KEY_TYPE,VAL_TYPE>(buckets.data(),_mapInfo->sizePower),
             s);
         

         if (nOverflownElements ==0 ){
            hipFreeAsync(overflownElements,s);
            return ;
         }
         //If we do have overflown elements we put them back in the buckets
         hipStreamSynchronize(s);
         Hashers::reset_to_empty<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction>
                                 <<<nOverflownElements,defaults::MAX_BLOCKSIZE,0,s>>>
                                 (overflownElements,
                                  buckets.data(),
                                  _mapInfo->sizePower,
                                  _mapInfo->currentMaxBucketOverflow,
                                  nOverflownElements);
         _mapInfo->fill-=nOverflownElements;
         hipStreamSynchronize(s);

         DeviceHasher::insert(overflownElements,
                              buckets.data(),
                              _mapInfo->sizePower,
                              _mapInfo->currentMaxBucketOverflow,
                              &_mapInfo->currentMaxBucketOverflow,
                              &_mapInfo->fill,
                              nOverflownElements,&_mapInfo->err,s);

         hipFreeAsync(overflownElements,s);
         return ;
      }

      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(KEY_TYPE* keys,VAL_TYPE* vals,size_t len,float targetLF=0.5,hipStream_t s=0){
         //Here we do some calculations to estimate how much if any we need to grow our buckets
         size_t neededPowerSize=std::ceil(std::log2((_mapInfo->fill+len)*(1.0/targetLF)));
         if (neededPowerSize>_mapInfo->sizePower){
            resize(neededPowerSize,targets::device,s);
         }
         buckets.optimizeGPU(s);
         _mapInfo->currentMaxBucketOverflow=_mapInfo->currentMaxBucketOverflow;
         DeviceHasher::insert(keys,vals,buckets.data(),_mapInfo->sizePower,_mapInfo->currentMaxBucketOverflow,&_mapInfo->currentMaxBucketOverflow,&_mapInfo->fill,len,&_mapInfo->err,s);
         return;
      }

     
      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(hash_pair<KEY_TYPE,VAL_TYPE>* src, size_t len,float targetLF=0.5,hipStream_t s=0){
         if(len==0){
            set_status(status::success);
            return;
         }
         //Here we do some calculations to estimate how much if any we need to grow our buckets
         size_t neededPowerSize=std::ceil(std::log2(((_mapInfo->fill)+len)*(1.0/targetLF)));
         if (neededPowerSize>_mapInfo->sizePower){
            resize(neededPowerSize,targets::device,s);
         }
         buckets.optimizeGPU(s);
         DeviceHasher::insert(src,buckets.data(),_mapInfo->sizePower,_mapInfo->currentMaxBucketOverflow,&_mapInfo->currentMaxBucketOverflow,&_mapInfo->fill,len,&_mapInfo->err,s);
         return;
      }

      //Uses Hasher's retrieve_kernel to read all elements
      HASHINATOR_HOSTONLY
      void retrieve(KEY_TYPE* keys,VAL_TYPE* vals,size_t len,hipStream_t s=0){
         buckets.optimizeGPU(s);
         DeviceHasher::retrieve(keys,vals,buckets.data(),_mapInfo->sizePower,_mapInfo->currentMaxBucketOverflow,len,s);
         return;
      }

      //Uses Hasher's erase_kernel to delete all elements
      HASHINATOR_HOSTONLY
      void erase(KEY_TYPE* keys,size_t len,hipStream_t s=0){
         buckets.optimizeGPU(s);
         //Remember the last numeber of tombstones
         size_t tbStore=tombstone_count();
         DeviceHasher::erase(keys,buckets.data(),&_mapInfo->tombstoneCounter,_mapInfo->sizePower,_mapInfo->currentMaxBucketOverflow,len,s);
         size_t tombstonesAdded=tombstone_count()-tbStore;
         //Fill should be decremented by the number of tombstones added;
         _mapInfo->fill-=tombstonesAdded;
         return;
      }

      /**
       * Host function  that returns a device pointer that can be passed to CUDA kernels
       * The pointer is internally cleaned up by the destructors, however the user **must**
       * call download() after usage on device.
       */
      HASHINATOR_HOSTONLY
      Hashmap* upload(hipStream_t stream = 0 ){
         optimizeGPU(stream);
         hipMemcpyAsync(device_map, this, sizeof(Hashmap),hipMemcpyHostToDevice,stream);
         return device_map;
      }

      HASHINATOR_HOSTONLY
      void optimizeGPU(hipStream_t stream=0) noexcept{
         int device;
         hipGetDevice(&device);
         hipMemPrefetchAsync(_mapInfo ,sizeof(MapInfo),device,stream);
         buckets.optimizeGPU(stream);
      }

      /*Manually prefetch data on Host*/
      HASHINATOR_HOSTONLY
      void optimizeCPU(hipStream_t stream = 0)noexcept{
         hipMemPrefetchAsync(_mapInfo ,sizeof(MapInfo),hipCpuDeviceId,stream);
         buckets.optimizeCPU(stream);
      }

      HASHINATOR_HOSTONLY 
      void streamAttach(hipStream_t s, uint32_t  flags=hipMemAttachSingle){
         buckets.streamAttach(s,flags);
         hipStreamAttachMemAsync( s,(void*)_mapInfo, sizeof(MapInfo),flags );
         CheckErrors("Stream Attach");
         return;
      }

      //Just return the device pointer. Upload should be called fist 
      //othewise map bookeepping info will not be updated on device.
      HASHINATOR_HOSTONLY
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
      HASHINATOR_HOSTONLY
      void download(hipStream_t stream = 0){
         //Copy over fill as it might have changed
         optimizeCPU(stream);
         if (_mapInfo->currentMaxBucketOverflow>Hashinator::defaults::BUCKET_OVERFLOW){
            std::cout<<"Device Overflow"<<std::endl;
            rehash(_mapInfo->sizePower+1);
         }else{
            if(tombstone_count()>0){
               std::cout<<"Cleaning Tombstones"<<std::endl;
               clean_tombstones();
            }
         }
      }
      
      // Device Iterator type. Iterates through all non-empty buckets.
      class device_iterator  {
      private:
         size_t index;
         Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
      public:
         HASHINATOR_DEVICEONLY
         device_iterator(Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
         
         HASHINATOR_DEVICEONLY
         size_t getIndex() { return index; }
        
         HASHINATOR_DEVICEONLY
         device_iterator& operator++() {
            index++;
            while(index < hashtable->buckets.size()){
               if (hashtable->buckets[index].first != EMPTYBUCKET && hashtable->buckets[index].first != TOMBSTONE){
                  break;
               }
               index++;
            }
            return *this;
         }
         
         HASHINATOR_DEVICEONLY
         device_iterator operator++(int){
            device_iterator temp = *this;
            ++(*this);
            return temp;
         }
         
         HASHINATOR_DEVICEONLY
         bool operator==(device_iterator other) const {
            return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
         }
         HASHINATOR_DEVICEONLY
         bool operator!=(device_iterator other) const {
            return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
         }
         
         HASHINATOR_DEVICEONLY
         hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_DEVICEONLY
         hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }

      };


      class const_device_iterator  {
      private:
         size_t index;
         const Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
      public:
         HASHINATOR_DEVICEONLY
         explicit const_device_iterator(const Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}
         
         HASHINATOR_DEVICEONLY
         size_t getIndex() { return index; }
        
         HASHINATOR_DEVICEONLY
         const_device_iterator& operator++() {
            index++;
            while(index < hashtable->buckets.size()){
               if (hashtable->buckets[index].first != EMPTYBUCKET && hashtable->buckets[index].first != TOMBSTONE ){
                  break;
               }
               index++;
            }
            return *this;
         }
         
         HASHINATOR_DEVICEONLY
         const_device_iterator operator++(int){
            const_device_iterator temp = *this;
            ++(*this);
            return temp;
         }
         
         HASHINATOR_DEVICEONLY
         bool operator==(const_device_iterator other) const {
            return &hashtable->buckets[index] == &other.hashtable->buckets[other.index];
         }
         HASHINATOR_DEVICEONLY
         bool operator!=(const_device_iterator other) const {
            return &hashtable->buckets[index] != &other.hashtable->buckets[other.index];
         }
         
         HASHINATOR_DEVICEONLY
         const hash_pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_DEVICEONLY
         const hash_pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
      };


      // Element access by iterator
      HASHINATOR_DEVICEONLY 
      device_iterator device_find(KEY_TYPE key) {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

            if (candidate.first==TOMBSTONE){continue;}

            if (candidate.first == key) {
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
      const const_device_iterator device_find(KEY_TYPE key)const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
            const hash_pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

            if (candidate.first==TOMBSTONE){continue;}

            if (candidate.first == key) {
               // Found a match, return that
               return const_device_iterator(*this, (hashIndex + i) & bitMask);
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
      device_iterator device_end() { return device_iterator(*this, buckets.size()); }

      HASHINATOR_DEVICEONLY
      const_device_iterator device_end()const  { return const_device_iterator(*this, buckets.size()); }

      HASHINATOR_DEVICEONLY
      device_iterator device_begin() {
         for (size_t i = 0; i < buckets.size(); i++) {
            if (buckets[i].first != EMPTYBUCKET && buckets[i].first != TOMBSTONE) {
               return device_iterator(*this, i);
            }
         }
         return device_end();
      }

      HASHINATOR_DEVICEONLY
      const_device_iterator device_begin() const {
         for (size_t i = 0; i < buckets.size(); i++) {
            if (buckets[i].first != EMPTYBUCKET && buckets[i].first!=TOMBSTONE) {
               return const_device_iterator(*this, i);
            }
         }
         return device_end();
      }

      HASHINATOR_DEVICEONLY
      size_t device_erase(const KEY_TYPE& key) {
         device_iterator element = device_find(key);
         if(element == device_end()) {
            return 0;
         } else {
            device_erase(element);
            return 1;
         }
      }

      HASHINATOR_DEVICEONLY
      size_t device_count(const KEY_TYPE& key) const {
         if (device_find(key) != device_end()) {
            return 1;
         } else {
            return 0;
         }
      }
       
      //Remove with tombstones on device
      HASHINATOR_DEVICEONLY
      device_iterator device_erase(device_iterator keyPos){

         //Get the index of this entry
         size_t index = keyPos.getIndex();
         
         //If this is an empty bucket or a tombstone we can return already
         //TODO Use CAS here for safety
         KEY_TYPE& item=buckets[index].first;
         if (item==EMPTYBUCKET || item==TOMBSTONE){return ++keyPos;}

         //Let's simply add a tombstone here
         atomicExch(&buckets[index].first,TOMBSTONE);
         atomicSub((unsigned int*)(&_mapInfo->fill), 1);
         atomicAdd((unsigned int*)(&_mapInfo->tombstoneCounter), 1);
         ++keyPos;
         return keyPos;
      }

      private:
      /**Device code for inserting elements. Nonexistent elements get created.
         Tombstones are accounted for.
       */
      HASHINATOR_DEVICEONLY
      void insert_element(const KEY_TYPE& key,VAL_TYPE value, size_t &thread_overflowLookup) {
         int bitMask = (1 <<_mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);
         size_t i =0;
         while(i<buckets.size()){
            uint32_t vecindex=(hashIndex + i) & bitMask;
            KEY_TYPE old = atomicCAS(&buckets[vecindex].first, EMPTYBUCKET, key);
            //Key does not exist so we create it and incerement fill
            if (old == EMPTYBUCKET){
               atomicExch(&buckets[vecindex].first,key);
               atomicExch(&buckets[vecindex].second,value);
               atomicAdd((unsigned int*)(&_mapInfo->fill), 1);
               thread_overflowLookup = i+1;
               return;
            }

            //Key exists so we overwrite it. Fill stays the same
            if (old == key){
               atomicExch(&buckets[vecindex].second,value);
               thread_overflowLookup = i+1;
               return;
            }

            i++;
         }
         assert(false && "Hashmap completely overflown");
      }

      public:
      HASHINATOR_DEVICEONLY
      hash_pair<device_iterator, bool> device_insert(hash_pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = device_find(newEntry.first) != device_end();
         if (!found) {
            set_element(newEntry.first,newEntry.second);
         }
         return hash_pair<device_iterator, bool>(device_find(newEntry.first), !found);
      }

      HASHINATOR_DEVICEONLY
      void set_element(const KEY_TYPE& key,VAL_TYPE val){
         size_t thread_overflowLookup;
         insert_element(key,val,thread_overflowLookup);
         atomicMax((unsigned long long*)&(_mapInfo->currentMaxBucketOverflow),nextPow2(thread_overflowLookup));
      }

      HASHINATOR_DEVICEONLY
      const VAL_TYPE& read_element(const KEY_TYPE& key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++) {
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
      }

      
      #else

      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(KEY_TYPE* keys,VAL_TYPE* vals,size_t len,float targetLF=0.5){
         for (size_t i=0; i<len; ++i){
            _at(keys[i])=vals[i];
         }

      }
     
      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(hash_pair<KEY_TYPE,VAL_TYPE>* src, size_t len,float targetLF=0.5){
         for (size_t i=0; i<len; ++i){
            _at(src[i].first)=src[i].second;
         }
      }

      //Uses Hasher's retrieve_kernel to read all elements
      HASHINATOR_HOSTONLY
      void retrieve(KEY_TYPE* keys,VAL_TYPE* vals,size_t len){
         for (size_t i=0; i<len; ++i){
            vals[i]=at(keys[i]);
         }
      }

      //Uses Hasher's erase_kernel to delete all elements
      HASHINATOR_HOSTONLY
      void erase(KEY_TYPE* keys,VAL_TYPE* vals,size_t len){
         for (size_t i=0; i<len;++i){
            erase(keys[i].first);
         }
      }
      
      #endif
   };
}//namespace Hashinator

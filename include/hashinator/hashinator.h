/* File:    hashinator.h
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
#include "../common.h"
#include "../splitvector/splitvec.h"
#include "../splitvector/split_allocators.h"
#include "hashfunctions.h"
#include "defaults.h"
#include <cuda/std/utility>
#ifndef HASHINATOR_HOST_ONLY
#include "../splitvector/split_tools.h"
#include "hashers.h"
#endif


namespace Hashinator{

   #ifndef HASHINATOR_HOST_ONLY
   template <typename T>
   using DefaultMetaAllocator = split::split_unified_allocator<T>;
   #define DefaultHasher Hashers::Hasher<KEY_TYPE,VAL_TYPE,HashFunction,EMPTYBUCKET,defaults::WARPSIZE,defaults::elementsPerWarp>
   #else
   template <typename T>
   using DefaultMetaAllocator = split::split_host_allocator<T>;
   #define DefaultHasher int //ugly TOFIX TODO
   #endif

   typedef struct Info {
      Info(int sz)
         :sizePower(sz),fill(0),cpu_maxBucketOverflow(0),postDevice_maxBucketOverflow(0),tombstoneCounter(0){}
      int sizePower;
      size_t fill; 
      int cpu_maxBucketOverflow;
      int postDevice_maxBucketOverflow;
      size_t tombstoneCounter;
   }MapInfo;

   template <typename KEY_TYPE, 
             typename VAL_TYPE, 
             int maxBucketOverflow = Hashinator::defaults::BUCKET_OVERFLOW, 
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
      split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>> buckets;
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
      // This helps in reducing th;e number of calls to cudaMalloc
      HASHINATOR_HOSTONLY
      void preallocate_device_handles(){
         #ifndef HASHINATOR_HOST_ONLY
         cudaMalloc((void **)&device_map, sizeof(Hashmap));
         #endif
      }

      // Deallocates the bookeepping info and the device pointer
      HASHINATOR_HOSTONLY
      void deallocate_device_handles(){
         #ifndef HASHINATOR_HOST_ONLY
         cudaFree(device_map);
         #endif
      }

   public:
      template <typename T, typename U>
      struct Overflown_Predicate{

      cuda::std::pair<KEY_TYPE, VAL_TYPE> *bck_ptr;
      int currentSizePower;
   
      explicit Overflown_Predicate(cuda::std::pair<KEY_TYPE, VAL_TYPE>*ptr,int s):bck_ptr(ptr),currentSizePower(s){}
      Overflown_Predicate()=delete;
         HASHINATOR_HOSTDEVICE
         inline bool operator()( cuda::std::pair<T,U>& element)const{
            if (element.first==TOMBSTONE){element.first=EMPTYBUCKET;return false;}
            if (element.first==EMPTYBUCKET){return false;}
            const size_t hashIndex = HashFunction::_hash(element.first,currentSizePower);
            const int bitMask = (1 <<(currentSizePower )) - 1; 
            bool isOverflown=(bck_ptr[hashIndex&bitMask].first!=(int)element.first);
            return isOverflown;
         }
      };
      HASHINATOR_HOSTONLY
      Hashmap(){
         preallocate_device_handles();
         _mapInfo=_metaAllocator.allocate(1);
         *_mapInfo=MapInfo(5);
         buckets=split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>> (1 << _mapInfo->sizePower, cuda::std::pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
       };

      HASHINATOR_HOSTONLY
      Hashmap(int sizepower){
         preallocate_device_handles();
         _mapInfo=_metaAllocator.allocate(1);
         *_mapInfo=MapInfo(sizepower);
         buckets=split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>> (1 << _mapInfo->sizePower, cuda::std::pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
       };

      HASHINATOR_HOSTONLY
      Hashmap(const Hashmap<KEY_TYPE, VAL_TYPE>& other){
            preallocate_device_handles();
            std::swap(_mapInfo, other._mapInfo);
            buckets=other.buckets;
       };

      HASHINATOR_HOSTONLY
      ~Hashmap(){     
         deallocate_device_handles();
         _metaAllocator.deallocate(_mapInfo,1);
      };


      // Resize the table to fit more things. This is automatically invoked once
      // maxBucketOverflow has triggered. This can only be done on host (so far)
      HASHINATOR_HOSTONLY
      void rehash(int newSizePower) {
         if (newSizePower > 32) {
            throw std::out_of_range("Hashmap ran into rehashing catastrophe and exceeded 32bit buckets.");
         }
         split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>> newBuckets(1 << newSizePower,
                                                     cuda::std::pair<KEY_TYPE, VAL_TYPE>(EMPTYBUCKET, VAL_TYPE()));
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
            for (int i = 0; i < maxBucketOverflow; i++) {
               cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = newBuckets[(newHash + i) & bitMask];
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
      }

      // Element access (by reference). Nonexistent elements get created.
      HASHINATOR_HOSTONLY
      VAL_TYPE& _at(const KEY_TYPE& key) {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
         }

         // Not found, and we have no free slots to create a new one. So we need to rehash to a larger size.
         rehash(_mapInfo->sizePower + 1);
         return at(key); // Recursive tail call to try again with larger table.
      }

      HASHINATOR_HOSTONLY
      const VAL_TYPE& _at(const KEY_TYPE& key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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

      HASHINATOR_HOSTONLY
      void clear() {
         buckets = split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>>(1 << _mapInfo->sizePower, {EMPTYBUCKET, VAL_TYPE()});
         _mapInfo->fill = 0;
      }

      //Try to grow our buckets until we achieve a targetLF load factor
      HASHINATOR_HOSTONLY
      void resize_to_lf(float targetLF=0.5){
         while (load_factor() > targetLF){
            rehash(_mapInfo->sizePower+1);
         }
      }

      HASHINATOR_HOSTONLY
      void resize(int newSizePower){
         rehash(newSizePower);     
      }



      HASHINATOR_HOSTONLY
         void print_pair(const cuda::std::pair<KEY_TYPE, VAL_TYPE>& i)const {
            if (i.first==TOMBSTONE){
               std::cout<<"[╀,-,-] ";
            }else if (i.first == EMPTYBUCKET){
               std::cout<<"[▢,-,-] ";
            }
            else{
               printf("[%d,%d] ",i.first,i.second);
            }
         }
      HASHINATOR_HOSTONLY
      void dump_buckets()const {
         std::cout<<_mapInfo->fill<<" "<<load_factor()<<std::endl;
         std::cout<<"\n";
         for  (int i =0 ; i < buckets.size(); ++i){
            print_pair(buckets[i]);
         }
         std::cout<<std::endl;

      }
       HASHINATOR_HOSTONLY
      size_t tombstone_count()const {
         return _mapInfo->tombstoneCounter;
      }

      HASHINATOR_HOSTONLY
      void swap(Hashmap<KEY_TYPE, VAL_TYPE>& other) noexcept{
         buckets.swap(other.buckets);
         std::swap(_mapInfo,other._mapInfo);
         std::swap(device_map,other.device_map);
      }

      //Read only  access to reference. 
      HASHINATOR_HOSTONLY
      const VAL_TYPE& at(const KEY_TYPE& key) const {
         return _at(key);
      }

      //See _at(key)
      HASHINATOR_HOSTONLY
      VAL_TYPE& at(const KEY_TYPE& key) {
         return _at(key);
      }

      // Typical array-like access with [] operator
      HASHINATOR_HOSTONLY
      VAL_TYPE& operator[](const KEY_TYPE& key) {
         return at(key); 
      }
      
      
      // Iterator type. Iterates through all non-empty buckets.
      class iterator : public std::iterator<std::random_access_iterator_tag, cuda::std::pair<KEY_TYPE, VAL_TYPE>> {
         Hashmap<KEY_TYPE, VAL_TYPE>* hashtable;
         size_t index;

      public:
         HASHINATOR_HOSTONLY
         iterator(Hashmap<KEY_TYPE, VAL_TYPE>& hashtable, size_t index) : hashtable(&hashtable), index(index) {}

         HASHINATOR_HOSTONLY
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
         cuda::std::pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         cuda::std::pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         size_t getIndex() { return index; }
      };

      // Const iterator.
      class const_iterator : public std::iterator<std::random_access_iterator_tag, cuda::std::pair<KEY_TYPE, VAL_TYPE>> {
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
               if (hashtable->buckets[index].first != EMPTYBUCKET){
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
         const cuda::std::pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         const cuda::std::pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
         HASHINATOR_HOSTONLY
         size_t getIndex() { return index; }
      };

      // Element access by iterator
      HASHINATOR_HOSTONLY
      const const_iterator find(KEY_TYPE key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < maxBucketOverflow; i++) {
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];
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
            if (buckets[i].first != EMPTYBUCKET) {
               return iterator(*this, i);
            }
         }
         return end();
      }

      HASHINATOR_HOSTONLY
      const_iterator begin() const {
         for (size_t i = 0; i < buckets.size(); i++) {
            if (buckets[i].first != EMPTYBUCKET) {
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
         // Due to overflowing buckets, this might require moving quite a bit of stuff around.
         size_t index = keyPos.getIndex();

         if (buckets[index].first != EMPTYBUCKET) {
            // Decrease fill count
            _mapInfo->fill--;

            // Clear the element itself.
            buckets[index].first = EMPTYBUCKET;

            int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
            size_t targetPos = index;
            // Search ahead to verify items are in correct places (until empty bucket is found)
            for (unsigned int i = 1; i < _mapInfo->fill; i++) {
               KEY_TYPE nextBucket = buckets[(index + i)&bitMask].first;
               if (nextBucket == EMPTYBUCKET) {
                  // The next bucket is empty, we are done.
                  break;
               }
               // Found an entry: is it in the correct bucket?
               uint32_t hashIndex = hash(nextBucket);
               if ((hashIndex&bitMask) != ((index + i)&bitMask)) {
                  // This entry has overflown. Now check if it should be moved:
                  uint32_t distance =  ((targetPos - hashIndex + (1<<_mapInfo->sizePower) )&bitMask);
                  if (distance < maxBucketOverflow) {
                     // Copy this entry to the current newly empty bucket, then continue with deleting
                     // this overflown entry and continue searching for overflown entries
                     VAL_TYPE moveValue = buckets[(index+i)&bitMask].second;
                     buckets[targetPos] = cuda::std::pair<KEY_TYPE, VAL_TYPE>(nextBucket,moveValue);
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

      HASHINATOR_HOSTONLY
      cuda::std::pair<iterator, bool> insert(cuda::std::pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = find(newEntry.first) != end();
         if (!found) {
            at(newEntry.first) = newEntry.second;
         }
         return cuda::std::pair<iterator, bool>(find(newEntry.first), !found);
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


      //Cleans all tombstones using splitvectors stream compcation and
      //the member Hasher
      HASHINATOR_HOSTONLY
      void clean_tombstones(){
         //Reset the tomstone counter
         _mapInfo->tombstoneCounter=0;
         //Allocate memory for overflown elements. So far this is the same size as our buckets but we can be better than this 
         //TODO size of overflown elements is known beforhand.
         split::SplitVector<cuda::std::pair<KEY_TYPE, VAL_TYPE>> overflownElements(1 << _mapInfo->sizePower, {EMPTYBUCKET, VAL_TYPE()});
         //Extract all overflown elements-This also resets TOMSBTONES to EMPTYBUCKET!
         split::tools::copy_if<cuda::std::pair<KEY_TYPE, VAL_TYPE>,Overflown_Predicate<KEY_TYPE,VAL_TYPE>,32,defaults::WARPSIZE>(buckets,overflownElements,Overflown_Predicate<KEY_TYPE,VAL_TYPE>(buckets.data(),_mapInfo->sizePower));
         size_t nOverflownElements=overflownElements.size();
         if (nOverflownElements ==0 ){
            std::cout<<"No cleaning needed!"<<std::endl;
            return ;
         }
         //If we do have overflown elements we put them back in the buckets
         Hashers::reset_to_empty<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction><<<overflownElements.size(),defaults::MAX_BLOCKSIZE>>> (overflownElements.data(),buckets.data(),_mapInfo->sizePower,maxBucketOverflow,overflownElements.size());
         _mapInfo->fill-=overflownElements.size();
         cudaDeviceSynchronize();
         DeviceHasher::insert(overflownElements.data(),buckets.data(),_mapInfo->sizePower,maxBucketOverflow,&_mapInfo->cpu_maxBucketOverflow,&_mapInfo->fill,overflownElements.size());
         if (_mapInfo->cpu_maxBucketOverflow>maxBucketOverflow){
            rehash(_mapInfo->sizePower++);
         }
         return ;
      }

      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(KEY_TYPE* keys,VAL_TYPE* vals,size_t len,float targetLF=0.5){
         //Here we do some calculations to estimate how much if any we need to grow our buckets
         size_t neededPowerSize=std::ceil(std::log2((_mapInfo->fill+len)*(1.0/targetLF)));
         if (neededPowerSize>_mapInfo->sizePower){
            resize(neededPowerSize);
         }
         buckets.optimizeGPU();
         _mapInfo->cpu_maxBucketOverflow=maxBucketOverflow;
         DeviceHasher::insert(keys,vals,buckets.data(),_mapInfo->sizePower,maxBucketOverflow,&_mapInfo->cpu_maxBucketOverflow,&_mapInfo->fill,len);
         if (_mapInfo->cpu_maxBucketOverflow>maxBucketOverflow){
            rehash(_mapInfo->sizePower++);
         }
         return;
      }

     
      //Uses Hasher's insert_kernel to insert all elements
      HASHINATOR_HOSTONLY
      void insert(cuda::std::pair<KEY_TYPE,VAL_TYPE>* src, size_t len,float targetLF=0.5){
         //Here we do some calculations to estimate how much if any we need to grow our buckets
         size_t neededPowerSize=std::ceil(std::log2(((_mapInfo->fill)+len)*(1.0/targetLF)));
         if (neededPowerSize>_mapInfo->sizePower){
            resize(neededPowerSize);
         }
         buckets.optimizeGPU();
         _mapInfo->cpu_maxBucketOverflow=maxBucketOverflow;
         DeviceHasher::insert(src,buckets.data(),_mapInfo->sizePower,maxBucketOverflow,&_mapInfo->cpu_maxBucketOverflow,&_mapInfo->fill,len);
         if (_mapInfo->cpu_maxBucketOverflow>maxBucketOverflow){
            rehash(_mapInfo->sizePower++);
         }
         return;
      }

      //Uses Hasher's retrieve_kernel to read all elements
      HASHINATOR_HOSTONLY
      void retrieve(KEY_TYPE* keys,VAL_TYPE* vals,size_t len){
         buckets.optimizeGPU();
         DeviceHasher::retrieve(keys,vals,buckets.data(),_mapInfo->sizePower,maxBucketOverflow,len);
         return;
      }

      //Uses Hasher's erase_kernel to delete all elements
      HASHINATOR_HOSTONLY
      void erase(KEY_TYPE* keys,VAL_TYPE* vals,size_t len){
         buckets.optimizeGPU();
         DeviceHasher::erase(keys,vals,buckets.data(),_mapInfo->tombstoneCounter,_mapInfo->sizePower,maxBucketOverflow,len);
         if (tombstone_count()>0){
            clean_tombstones();
         }
         return;
      }

      /**
       * Host function  that returns a device pointer that can be passed to CUDA kernels
       * The pointer is internally cleaned up by the destructors, however the user **must**
       * call download() after usage on device.
       */
      HASHINATOR_HOSTONLY
      Hashmap* upload(cudaStream_t stream = 0 ){
         _mapInfo->cpu_maxBucketOverflow=maxBucketOverflow;
         optimizeGPU(stream);
         cudaMemcpyAsync(device_map, this, sizeof(Hashmap),cudaMemcpyHostToDevice,stream);
         return device_map;
      }

      HASHINATOR_HOSTONLY
      void optimizeGPU(cudaStream_t stream=0) noexcept{
         int device;
         cudaGetDevice(&device);
         cudaMemPrefetchAsync(_mapInfo ,sizeof(MapInfo),device,stream);
         buckets.optimizeGPU(stream);
      }

      /*Manually prefetch data on Host*/
      HASHINATOR_HOSTONLY
      void optimizeCPU(cudaStream_t stream = 0)noexcept{
         cudaMemPrefetchAsync(_mapInfo ,sizeof(MapInfo),cudaCpuDeviceId,stream);
         buckets.optimizeCPU(stream);
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
      void download(cudaStream_t stream = 0){
         //Copy over fill as it might have changed
         optimizeCPU(stream);
         if (_mapInfo->cpu_maxBucketOverflow>maxBucketOverflow){
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
               if (hashtable->buckets[index].first != EMPTYBUCKET&&
                   hashtable->buckets[index].first != TOMBSTONE){
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
         cuda::std::pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_DEVICEONLY
         cuda::std::pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }

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
               if (hashtable->buckets[index].first != EMPTYBUCKET &&
                   hashtable->buckets[index].first != TOMBSTONE ){
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
         const cuda::std::pair<KEY_TYPE, VAL_TYPE>& operator*() const { return hashtable->buckets[index]; }
         HASHINATOR_DEVICEONLY
         const cuda::std::pair<KEY_TYPE, VAL_TYPE>* operator->() const { return &hashtable->buckets[index]; }
      };


      // Element access by iterator
      HASHINATOR_DEVICEONLY 
      device_iterator device_find(KEY_TYPE key) {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < _mapInfo->cpu_maxBucketOverflow; i++) {
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

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
         for (int i = 0; i < _mapInfo->cpu_maxBucketOverflow; i++) {
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[(hashIndex + i) & bitMask];

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
            if (buckets[i].first != EMPTYBUCKET) {
               return device_iterator(*this, i);
            }
         }
         return device_end();
      }


      HASHINATOR_DEVICEONLY
      size_t device_erase(const KEY_TYPE& key) {
         iterator element = device_find(key);
         if(element == device_end()) {
            return 0;
         } else {
            device_erase(element);
            return 1;
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
                     atomicAdd((unsigned int*)(&_mapInfo->fill), 1);
                     thread_overflowLookup = i+1;
                     return;
                  }
               }

            }
            i++;
         }
         assert(false && "Hashmap completely overflown");
      }

      HASHINATOR_DEVICEONLY
      cuda::std::pair<device_iterator, bool> device_insert(cuda::std::pair<KEY_TYPE, VAL_TYPE> newEntry) {
         bool found = device_find(newEntry.first) != device_end();
         if (!found) {
            set_element(newEntry.first,newEntry.second);
         }
         return cuda::std::pair<iterator, bool>(device_find(newEntry.first), !found);
      }

      HASHINATOR_DEVICEONLY
      void set_element(const KEY_TYPE& key,VAL_TYPE val){
         size_t thread_overflowLookup;
         insert_element(key,val,thread_overflowLookup);
         atomicMax(&(_mapInfo->cpu_maxBucketOverflow),thread_overflowLookup);
      }

      HASHINATOR_DEVICEONLY
      const VAL_TYPE& read_element(const KEY_TYPE& key) const {
         int bitMask = (1 << _mapInfo->sizePower) - 1; // For efficient modulo of the array size
         uint32_t hashIndex = hash(key);

         // Try to find the matching bucket.
         for (int i = 0; i < _mapInfo->cpu_maxBucketOverflow; i++) {
            uint32_t vecindex=(hashIndex + i) & bitMask;
            const cuda::std::pair<KEY_TYPE, VAL_TYPE>& candidate = buckets[vecindex];
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
      void insert(cuda::std::pair<KEY_TYPE,VAL_TYPE>* src, size_t len,float targetLF=0.5){
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

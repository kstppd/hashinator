/* File:    hashers.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 * Description: Defines parallel hashers that insert,retrieve and
 *               delete elements to/from Hahsinator on device
 *
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
#include "../common.h"
#include "../splitvector/gpu_wrappers.h"
#include "defaults.h"
#include "hashfunctions.h"
#ifdef __NVCC__
#include "kernels_NVIDIA.h"
#endif
#ifdef __HIP__
#include "kernels_AMD.h"
#endif

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

template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE ,
          KEY_TYPE , class HashFunction,
          class DeviceHasher , class Meta_Allocator>
class Hashmap;
  
namespace Hashers {
template <typename KEY_TYPE, typename VAL_TYPE, class HashFunction,
          KEY_TYPE, KEY_TYPE,
          int ,int >
class Hasher;

template <typename KEY_TYPE, typename VAL_TYPE, class HashFunction,
          KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(), KEY_TYPE TOMBSTONE = EMPTYBUCKET ,
          int WARP = defaults::WARPSIZE, int elementsPerWarp = 1>
class Hasher {

   // Make sure we have sane elements per warp
   static_assert(elementsPerWarp > 0 && elementsPerWarp <= WARP && "Device hasher cannot be instantiated");

public:
   // Overload with separate input for keys and values.
   static void insert(KEY_TYPE* keys, VAL_TYPE* vals, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                      Hashinator::Info* info, size_t len,
                      split_gpuStream_t s = 0) {
      size_t blocks, blockSize;
      info->err = status::success;
      launchParams(len, blocks, blockSize);
      Hashinator::Hashers::insert_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE,
                                         elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(keys, vals, buckets, info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
#ifndef NDEBUG
      if (info->err == status::fail) {
         std::cerr << "***** Hashinator Runtime Warning ********" << std::endl;
         std::cerr << "Warning: Hashmap completely overflown in Device Insert.\nNot all ellements were "
                      "inserted!\nConsider resizing before calling insert"
                   << std::endl;
         std::cerr << "******************************" << std::endl;
      }
#endif
   }

   // Overload with input for keys only, using the index as the value
   static void insertIndex(KEY_TYPE* keys, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, Hashinator::Info* info,
                           size_t len, split_gpuStream_t s = 0) {
      size_t blocks, blockSize;
      info->err = status::success;
      launchParams(len, blocks, blockSize);
      Hashinator::Hashers::insert_index_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE,
                                               elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(keys, buckets, info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
#ifndef NDEBUG
      if (info->err == status::fail) {
         std::cerr << "***** Hashinator Runtime Warning ********" << std::endl;
         std::cerr << "Warning: Hashmap completely overflown in Device InsertIndex.\nNot all elements were "
                      "inserted!\nConsider resizing before calling insert"
                   << std::endl;
         std::cerr << "******************************" << std::endl;
      }
#endif
   }

   // Overload with hash_pair<key,val> (k,v) inputs
   // Used by the tombstone cleaning method.
   static void insert(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                      Hashinator::Info* info, size_t len, split_gpuStream_t s = 0) {
      size_t blocks, blockSize;
      info->err = status::success;
      launchParams(len, blocks, blockSize);
      Hashinator::Hashers::insert_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE,
                                         elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(src, buckets, info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
#ifndef NDEBUG
      if (info->err == status::fail) {
         std::cerr << "***** Hashinator Runtime Warning ********" << std::endl;
         std::cerr << "Warning: Hashmap completely overflown in Device Insert.\nNot all ellements were "
                      "inserted!\nConsider resizing before calling insert"
                   << std::endl;
         std::cerr << "******************************" << std::endl;
      }
#endif
   }

   // Retrieve wrapper
   static void retrieve(KEY_TYPE* keys, VAL_TYPE* vals, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                        Hashinator::Info* info, size_t len, split_gpuStream_t s = 0) {

      size_t blocks, blockSize;
      launchParams(len, blocks, blockSize);
      retrieve_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE, elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(keys, vals, buckets, info);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }

   static void retrieve(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                        Hashinator::Info* info, size_t len, split_gpuStream_t s = 0) {

      size_t blocks, blockSize;
      launchParams(len, blocks, blockSize);
      retrieve_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE, elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(src, buckets, info);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }

   // Delete wrapper
   static void erase(KEY_TYPE* keys, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                     Hashinator::Info* info, size_t len, split_gpuStream_t s = 0) {

      size_t blocks, blockSize;
      launchParams(len, blocks, blockSize);
      Hashinator::Hashers::delete_kernel<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, TOMBSTONE, HashFunction, defaults::WARPSIZE,
                                         elementsPerWarp>
         <<<blocks, blockSize, 0, s>>>(keys, buckets, info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }

   // Reset wrapper
   static void reset(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* dst,
                     Hashinator::Info* info, size_t len, split_gpuStream_t s = 0) {
      size_t blocks, blockSize;
      launchParams(len, blocks, blockSize);
      Hashinator::Hashers::reset_to_empty<KEY_TYPE, VAL_TYPE, EMPTYBUCKET, HashFunction, defaults::WARPSIZE,
                                          elementsPerWarp>
          <<<blocks, blockSize, 0, s>>>(src, dst, info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }

   template <class U,class UU , class UUU>
   void sudo_ingest(Hashinator::Hashmap<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,TOMBSTONE,U,UU,UUU> **ptr){
     
   }

   // Reset wrapper for all elements
   static void reset_all(hash_pair<KEY_TYPE, VAL_TYPE>* dst, Hashinator::Info* info, size_t len,
                         split_gpuStream_t s = 0) {
      // fast ceil for positive ints
      size_t blocksNeeded = len / defaults::MAX_BLOCKSIZE + (len % defaults::MAX_BLOCKSIZE != 0);
      reset_all_to_empty<KEY_TYPE, VAL_TYPE, EMPTYBUCKET>
         <<<blocksNeeded, defaults::MAX_BLOCKSIZE, 0, s>>>(dst,info, len);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }

private:
   static void launchParams(size_t N, size_t& blocks, size_t& blockSize) {
      // fast ceil for positive ints
      size_t warpsNeeded = N / elementsPerWarp + (N % elementsPerWarp != 0);
      blockSize = std::min(warpsNeeded * WARP, static_cast<size_t>(defaults::MAX_BLOCKSIZE));
      blocks = warpsNeeded * WARP / blockSize + ((warpsNeeded * WARP) % blockSize != 0);
      return;
   }
};

} // namespace Hashers
} // namespace Hashinator

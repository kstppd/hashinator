/* File:    kernels_NVIDIA.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
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

namespace Hashinator {
namespace Hashers {

/*
 * Resets all elements in dst to EMPTY, VAL_TYPE()
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max()>
__global__ void reset_all_to_empty(hash_pair<KEY_TYPE, VAL_TYPE>* dst, const size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   // Early exit here
   if (tid >= len) {
      return;
   }

   if (dst[tid].first != EMPTYBUCKET) {
      dst[tid].first = EMPTYBUCKET;
   }
   return;
}

template <int WARPSIZE>
HASHINATOR_DEVICEONLY __forceinline__ int warpReduce(int localCount) {
   for (int i = WARPSIZE / 2; i > 0; i = i / 2) {
      localCount += split::s_shuffle_down(localCount, i, SPLIT_VOTING_MASK);
   }
   return localCount;
}

template <int WARPSIZE>
HASHINATOR_DEVICEONLY __forceinline__ uint64_t warpReduceMax(uint64_t entry) {
   for (int i = WARPSIZE / 2; i > 0; i = i / 2) {
      entry = std::max(entry, split::s_shuffle_down(entry, i, SPLIT_VOTING_MASK));
   }
   return entry;
}

/*
 * Resets all elements pointed by src to EMPTY in dst
 * If an elements in src is not found this will assert(false)
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void reset_to_empty(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* dst,
                               const int sizePower, size_t maxoverflow, size_t len)

{
   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;

   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   hash_pair<KEY_TYPE, VAL_TYPE> candidate = src[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidate.first, sizePower);
   uint64_t vWarpDone = 0; // state of virtual warp

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = dst[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target.first == candidate.first, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            dst[probingindex].first = EMPTYBUCKET;
            vWarpDone = 1;
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }
   return;
}

template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void insert_kernel(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, int sizePower,
                              size_t maxoverflow, size_t* d_overflow, size_t* d_fill, size_t len, status* err) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t addMask[WARPSIZE / 2];
   __shared__ uint64_t warpOverflow[WARPSIZE / 2];
   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         addMask[i] = 0;
         warpOverflow[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   hash_pair<KEY_TYPE, VAL_TYPE> candidate = src[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidate.first, sizePower);
   uint32_t localCount = 0;
   uint64_t threadOverflow = 0;
   uint64_t vWarpDone = 0; // state of virtual warp

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = buckets[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target.first == EMPTYBUCKET, submask) & submask;

      // Check if this elements already exists
      auto already_exists = split::s_warpVote(target.first == candidate.first, submask) & submask;
      if (already_exists) {
         int winner = split::s_findFirstSig(already_exists) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            split::s_atomicExch(&buckets[probingindex].second, candidate.second);
            // This virtual warp is now done.
            vWarpDone = 1;
         }
      }

      // If any duplicate was there now is the time for the whole Virtual warp to find out!
      vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
            if (old == EMPTYBUCKET) {
               threadOverflow = std::min(i + w_tid, static_cast<size_t>(1 << sizePower)) + 1;
               split::s_atomicExch(&buckets[probingindex].second, candidate.second);
               vWarpDone = 1;
               // Flip the bit which corresponds to the thread that added an element
               localCount++;
            } else if (old == candidate.first) {
               // Parallel stuff are fun. Major edge case!
               split::s_atomicExch(&buckets[probingindex].second, candidate.second);
               vWarpDone = 1;
            }
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }

   // Update fill and overflow
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   uint64_t perWarpOverflow = warpReduceMax<WARPSIZE>(threadOverflow);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      addMask[(blockWid)] = warpTotals;
      warpOverflow[(blockWid)] = perWarpOverflow;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      uint64_t blockOverflow = warpReduceMax<WARPSIZE / 2>(warpOverflow[(proper_w_tid)]);
      int blockTotal = warpReduce<WARPSIZE / 2>(addMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         if (blockOverflow > *d_overflow) {
            split::s_atomicExch((unsigned long long*)d_overflow,
                                (unsigned long long)nextOverflow(blockOverflow, VIRTUALWARP));
         }
         split::s_atomicAdd(d_fill, blockTotal);
      }
   }
   return;
}

template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void insert_kernel(KEY_TYPE* keys, VAL_TYPE* vals, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, int sizePower,
                              size_t maxoverflow, size_t* d_overflow, size_t* d_fill, size_t len, status* err) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t addMask[WARPSIZE / 2];
   __shared__ uint64_t warpOverflow[WARPSIZE / 2];
   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         addMask[i] = 0;
         warpOverflow[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidateKey = keys[wid];
   VAL_TYPE candidateVal = vals[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
   uint32_t localCount = 0;
   uint64_t vWarpDone = 0; // state of virtual warp
   uint64_t threadOverflow = 0;

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }
      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = buckets[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target.first == EMPTYBUCKET, submask) & submask;

      // Check if this elements already exists
      auto already_exists = split::s_warpVote(target.first == candidateKey, submask) & submask;
      if (already_exists) {
         int winner = split::s_findFirstSig(already_exists) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            split::s_atomicExch(&buckets[probingindex].second, candidateVal);
            // This virtual warp is now done.
            vWarpDone = 1;
         }
      }

      // If any duplicate was there now is the time for the whole Virtual warp to find out!
      vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidateKey);
            if (old == EMPTYBUCKET) {
               threadOverflow = std::min(i + w_tid, static_cast<size_t>(1 << sizePower)) + 1;
               split::s_atomicExch(&buckets[probingindex].second, candidateVal);
               vWarpDone = 1;
               // Flip the bit which corresponds to the thread that added an element
               localCount++;
            } else if (old == candidateKey) {
               // Parallel stuff are fun. Major edge case!
               split::s_atomicExch(&buckets[probingindex].second, candidateVal);
               vWarpDone = 1;
            }
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }

   // Update fill and overflow
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   uint64_t perWarpOverflow = warpReduceMax<WARPSIZE>(threadOverflow);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      addMask[(blockWid)] = warpTotals;
      warpOverflow[(blockWid)] = perWarpOverflow;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      uint64_t blockOverflow = warpReduceMax<WARPSIZE / 2>(warpOverflow[(proper_w_tid)]);
      int blockTotal = warpReduce<WARPSIZE / 2>(addMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         if (blockOverflow > *d_overflow) {
            split::s_atomicExch((unsigned long long*)d_overflow,
                                (unsigned long long)nextOverflow(blockOverflow, VIRTUALWARP));
         }
         split::s_atomicAdd(d_fill, blockTotal);
      }
   }
   return;
}

/*
 * In a similar way to the insert and retrieve kernels we
 * delete keys in "keys" if they do exist in the hasmap.
 * If the keys do not exist we do nothing.
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1, class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>,
          int WARPSIZE = defaults::WARPSIZE, int elementsPerWarp>
__global__ void delete_kernel(KEY_TYPE* keys, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, size_t* d_tombstoneCounter,
                              int sizePower, size_t maxoverflow, size_t len) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t deleteMask[WARPSIZE / 2];

   // Early quit if we have more warps than elements to handle
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         deleteMask[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidateKey = keys[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
   uint32_t localCount = 0;

   for (size_t i = 0; i < maxoverflow; i += VIRTUALWARP) {

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      const auto maskExists =
          split::s_warpVote(buckets[probingindex].first == candidateKey, SPLIT_VOTING_MASK) & submask;
      const auto emptyFound =
          split::s_warpVote(buckets[probingindex].first == EMPTYBUCKET, SPLIT_VOTING_MASK) & submask;
      // If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
      if (!maskExists && emptyFound) {
         return;
      }
      if (maskExists) {
         int winner = split::s_findFirstSig(maskExists) - 1;
         winner -= (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == winner) {
            split::s_atomicExch(&buckets[probingindex].first, TOMBSTONE);
            localCount++;
         }
         break;
      }
   }

   // Update tombstone counter
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      deleteMask[(blockWid)] = warpTotals;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      int blockTotal = warpReduce<WARPSIZE / 2>(deleteMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         split::s_atomicAdd(d_tombstoneCounter, blockTotal);
      }
   }
   return;
}

template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void insert_index_kernel(KEY_TYPE* keys, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, int sizePower,
                                    size_t maxoverflow, size_t* d_overflow, size_t* d_fill, size_t len, status* err) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t addMask[WARPSIZE / 2];
   __shared__ uint64_t warpOverflow[WARPSIZE / 2];
   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         addMask[i] = 0;
         warpOverflow[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidateKey = keys[wid];
   VAL_TYPE candidateVal = wid;
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
   uint32_t localCount = 0;
   uint64_t vWarpDone = 0; // state of virtual warp
   uint64_t threadOverflow = 0;

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }
      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = buckets[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target.first == EMPTYBUCKET, submask) & submask;

      // Check if this elements already exists
      auto already_exists = split::s_warpVote(target.first == candidateKey, submask) & submask;
      if (already_exists) {
         int winner = split::s_findFirstSig(already_exists) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            split::s_atomicExch(&buckets[probingindex].second, candidateVal);
            // This virtual warp is now done.
            vWarpDone = 1;
         }
      }

      // If any duplicate was there now is the time for the whole Virtual warp to find out!
      vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidateKey);
            if (old == EMPTYBUCKET) {
               threadOverflow = std::min(i + w_tid, static_cast<size_t>(1 << sizePower)) + 1;
               split::s_atomicExch(&buckets[probingindex].second, candidateVal);
               vWarpDone = 1;
               // Flip the bit which corresponds to the thread that added an element
               localCount++;
            } else if (old == candidateKey) {
               // Parallel stuff are fun. Major edge case!
               split::s_atomicExch(&buckets[probingindex].second, candidateVal);
               vWarpDone = 1;
            }
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }

   // Update fill and overflow
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   uint64_t perWarpOverflow = warpReduceMax<WARPSIZE>(threadOverflow);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      addMask[(blockWid)] = warpTotals;
      warpOverflow[(blockWid)] = perWarpOverflow;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      uint64_t blockOverflow = warpReduceMax<WARPSIZE / 2>(warpOverflow[(proper_w_tid)]);
      int blockTotal = warpReduce<WARPSIZE / 2>(addMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         if (blockOverflow > *d_overflow) {
            split::s_atomicExch((unsigned long long*)d_overflow,
                                (unsigned long long)nextOverflow(blockOverflow, VIRTUALWARP));
         }
         split::s_atomicAdd(d_fill, blockTotal);
      }
   }
   return;
}

/*
 * Similarly to the insert_kernel we examine elements in keys and return their value in vals,
 * if the do exist in the hashmap. If the elements is not found and invalid key is returned;
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void retrieve_kernel(KEY_TYPE* keys, VAL_TYPE* vals, hash_pair<KEY_TYPE, VAL_TYPE>* buckets, int sizePower,
                                size_t maxoverflow) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE& candidateKey = keys[wid];
   VAL_TYPE& candidateVal = vals[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);

   // Check for duplicates
   for (size_t i = 0; i < maxoverflow; i += VIRTUALWARP) {

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      const auto maskExists =
          split::s_warpVote(buckets[probingindex].first == candidateKey, SPLIT_VOTING_MASK) & submask;
      const auto emptyFound =
          split::s_warpVote(buckets[probingindex].first == EMPTYBUCKET, SPLIT_VOTING_MASK) & submask;
      // If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
      if (!maskExists && emptyFound) {
         return;
      }
      if (maskExists) {
         int winner = split::s_findFirstSig(maskExists) - 1;
         winner -= (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == winner) {
            split::s_atomicExch(&candidateVal, buckets[probingindex].second);
         }
         return;
      }
   }
}

/*
 * Similarly to the insert_kernel we examine elements in keys and return their value in vals,
 * if the do exist in the hashmap. If the elements is not found and invalid key is returned;
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void retrieve_kernel(hash_pair<KEY_TYPE, VAL_TYPE>* src, hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                                int sizePower, size_t maxoverflow) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   hash_pair<KEY_TYPE, VAL_TYPE>& candidate = src[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidate.first, sizePower);

   // Check for duplicates
   for (size_t i = 0; i < maxoverflow; i += VIRTUALWARP) {

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      const auto maskExists =
          split::s_warpVote(buckets[probingindex].first == candidate.first, SPLIT_VOTING_MASK) & submask;
      const auto emptyFound =
          split::s_warpVote(buckets[probingindex].first == EMPTYBUCKET, SPLIT_VOTING_MASK) & submask;
      // If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
      if (!maskExists && emptyFound) {
         return;
      }
      if (maskExists) {
         int winner = split::s_findFirstSig(maskExists) - 1;
         winner -= (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == winner) {
            split::s_atomicExch(&candidate.second, buckets[probingindex].second);
         }
         return;
      }
   }
}


/* ----------------------------------- Kernels used by Hashinator::Unordered_Set -----------------------------------*/

/*
 * Resets all elements in dst to EMPTY
 * */
template <typename KEY_TYPE, typename VAL_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max()>
__global__ void reset_all_to_empty_set(hash_pair<KEY_TYPE, VAL_TYPE>* dst, const size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   // Early exit here
   if (tid >= len) {
      return;
   }

   if (dst[tid] != EMPTYBUCKET) {
      dst[tid] = EMPTYBUCKET;
   }
   return;
}

template <typename KEY_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,int elementsPerWarp>
__global__ void insert_set_kernel(KEY_TYPE* keys, KEY_TYPE* buckets, int sizePower,size_t maxoverflow, size_t* d_overflow,
                              size_t* d_fill, size_t len, status* err) {

   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t addMask[WARPSIZE / 2];
   __shared__ uint64_t warpOverflow[WARPSIZE / 2];
   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         addMask[i] = 0;
         warpOverflow[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidateKey = keys[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
   uint32_t localCount = 0;
   uint64_t vWarpDone = 0; // state of virtual warp
   uint64_t threadOverflow = 0;

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }
      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = buckets[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target == EMPTYBUCKET, submask) & submask;

      // Check if this elements already exists
      auto already_exists = split::s_warpVote(target == candidateKey, submask) & submask;
      if (already_exists) {vWarpDone = 1;}

      // If any duplicate was there now is the time for the whole Virtual warp to find out!
      vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            KEY_TYPE old = split::s_atomicCAS(&buckets[probingindex], EMPTYBUCKET, candidateKey);
            if (old == EMPTYBUCKET) {
               threadOverflow = std::min(i + w_tid, static_cast<size_t>(1 << sizePower)) + 1;
               vWarpDone = 1;
               // Flip the bit which corresponds to the thread that added an element
               localCount++;
            } else if (old == candidateKey) {
               vWarpDone = 1;
            }
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }

   // Update fill and overflow
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   uint64_t perWarpOverflow = warpReduceMax<WARPSIZE>(threadOverflow);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      addMask[(blockWid)] = warpTotals;
      warpOverflow[(blockWid)] = perWarpOverflow;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      uint64_t blockOverflow = warpReduceMax<WARPSIZE / 2>(warpOverflow[(proper_w_tid)]);
      int blockTotal = warpReduce<WARPSIZE / 2>(addMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         if (blockOverflow > *d_overflow) {
            split::s_atomicExch((unsigned long long*)d_overflow,
                                (unsigned long long)nextOverflow(blockOverflow, VIRTUALWARP));
         }
         split::s_atomicAdd(d_fill, blockTotal);
      }
   }
   return;
}


/*
 * In a similar way to the insert and retrieve kernels we
 * delete keys in "keys" if they do exist in the set.
 * If the keys do not exist we do nothing.
 * */
template <typename KEY_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1, class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>,
          int WARPSIZE = defaults::WARPSIZE, int elementsPerWarp>
__global__ void delete_set_kernel(KEY_TYPE* keys, KEY_TYPE* buckets, size_t* d_tombstoneCounter,
                              int sizePower, size_t maxoverflow, size_t len) {


   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;
   const size_t proper_w_tid = tid % WARPSIZE; // the proper WID as if we had no Virtual warps
   const size_t proper_wid = tid / WARPSIZE;
   const size_t blockWid = proper_wid % (WARPSIZE / 4); // we have twice the warpsize and half the warps per block

   __shared__ uint32_t deleteMask[WARPSIZE / 2];

   // Early quit if we have more warps than elements to handle
   if (wid >= len) {
      return;
   }

   // Zero out shared count;
   if (proper_w_tid == 0 && blockWid == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
         deleteMask[i] = 0;
      }
   }
   __syncthreads();

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidateKey = keys[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidateKey, sizePower);
   uint32_t localCount = 0;

   for (size_t i = 0; i < maxoverflow; i += VIRTUALWARP) {

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      const auto maskExists =
          split::s_warpVote(buckets[probingindex] == candidateKey, SPLIT_VOTING_MASK) & submask;
      const auto emptyFound =
          split::s_warpVote(buckets[probingindex] == EMPTYBUCKET, SPLIT_VOTING_MASK) & submask;
      // If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
      if (!maskExists && emptyFound) {
         return;
      }
      if (maskExists) {
         int winner = split::s_findFirstSig(maskExists) - 1;
         winner -= (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == winner) {
            split::s_atomicExch(&buckets[probingindex], TOMBSTONE);
            localCount++;
         }
         break;
      }
   }

   // Update tombstone counter
   __syncthreads();
   // Per warp reduction
   int warpTotals = warpReduce<WARPSIZE>(localCount);
   __syncthreads();

   // Store to shmem minding Bank Conflicts
   if (proper_w_tid == 0) {
      // Write the count to the same place
      deleteMask[(blockWid)] = warpTotals;
   }

   __syncthreads();
   // First warp in block reductions
   if (blockWid == 0) {
      int blockTotal = warpReduce<WARPSIZE / 2>(deleteMask[(proper_w_tid)]);
      // First thread updates fill and overlfow (1 update per block)
      if (proper_w_tid == 0) {
         split::s_atomicAdd(d_tombstoneCounter, blockTotal);
      }
   }
   return;
}

/*
 * Resets all elements pointed by src to EMPTY in dst
 * If an elements in src is not found this will assert(false)
 * */
template <typename KEY_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>, int WARPSIZE = defaults::WARPSIZE,
          int elementsPerWarp>
__global__ void reset_to_empty_set(KEY_TYPE* src, KEY_TYPE* dst,const int sizePower, size_t maxoverflow, size_t len)


{
   const int VIRTUALWARP = WARPSIZE / elementsPerWarp;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t wid = tid / VIRTUALWARP;
   const size_t w_tid = tid % VIRTUALWARP;

   // Early quit if we have more warps than elements to insert
   if (wid >= len) {
      return;
   }

   uint64_t subwarp_relative_index = (wid) % (WARPSIZE / VIRTUALWARP);
   uint64_t submask;
   if constexpr (elementsPerWarp == 1) {
      // TODO mind AMD 64 thread wavefronts
      submask = SPLIT_VOTING_MASK;
   } else {
      submask = split::getIntraWarpMask_AMD(0, VIRTUALWARP * subwarp_relative_index + 1,
                                            VIRTUALWARP * subwarp_relative_index + VIRTUALWARP);
   }

   KEY_TYPE candidate = src[wid];
   const int bitMask = (1 << (sizePower)) - 1;
   const auto hashIndex = HashFunction::_hash(candidate, sizePower);
   uint64_t vWarpDone = 0; // state of virtual warp

   for (size_t i = 0; i < (1 << sizePower); i += VIRTUALWARP) {

      // Check if this virtual warp is done.
      if (vWarpDone) {
         break;
      }

      // Get the position we should be looking into
      size_t probingindex = ((hashIndex + i + w_tid) & bitMask);
      auto target = dst[probingindex];

      // vote for available emptybuckets in warp region
      // Note that this has to be done before voting for already existing elements (below)
      auto mask = split::s_warpVote(target == candidate, submask) & submask;

      while (mask && !vWarpDone) {
         int winner = split::s_findFirstSig(mask) - 1;
         int sub_winner = winner - (subwarp_relative_index)*VIRTUALWARP;
         if (w_tid == sub_winner) {
            dst[probingindex] = EMPTYBUCKET;
            vWarpDone = 1;
         }
         // If any of the virtual warp threads are done the the whole
         // Virtual warp is done
         vWarpDone = split::s_warpVote(vWarpDone > 0, submask) & submask;
         mask ^= (1UL << winner);
      }
   }
   return;
}

} // namespace Hashers
} // namespace Hashinator

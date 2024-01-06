/* File:    split::tools.h
 * Authors: Kostis Papadakis (2023)
 * Description: Set of tools used by SplitVector
 *
 * This file defines the following classes or functions:
 *    --split::tools::Cuda_mempool
 *    --split::tools::copy_if_raw
 *    --split::tools::copy_if
 *    --split::tools::scan_reduce_raw
 *    --split::tools::scan_reduce
 *    --split::tools::scan_add
 *    --split::tools::split_compact_raw
 *    --split::tools::split_compact_raw
 *    --split::tools::split_prefix_scan_raw
 *    --split::tools::split_prefix_scan
 *    --split::tools::split_prescan
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
#include "gpu_wrappers.h"
#ifndef SPLIT_CPU_ONLY_MODE
#include "devicevec.h"
#endif
#define NUM_BANKS 32 // TODO depends on device
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#ifdef __NVCC__
#define SPLIT_VOTING_MASK 0xFFFFFFFF // 32-bit wide for split_gpu warps
#define WARPLENGTH 32
#endif
#ifdef __HIP__
#define SPLIT_VOTING_MASK 0xFFFFFFFFFFFFFFFFull // 64-bit wide for amd warps
#define WARPLENGTH 64

#endif

namespace split {
namespace tools {

/**
 * @brief Check if a value is a power of two.
 *
 * This function checks whether the given value is a power of two.
 *
 * @param val The value to be checked.
 * @return constexpr inline bool Returns true if the value is a power of two, false otherwise.
 */
constexpr inline bool isPow2(const size_t val) noexcept { return (val & (val - 1)) == 0; }

/**
 * @brief GPU kernel for performing scan-add operation.
 *
 * This kernel performs a scan-add operation on the given input array, using the partial sums array.
 * It updates the input array with the computed values.
 * Part of splitvector's stream compaction mechanism.
 *
 * @tparam T Type of the array elements.
 * @param input The input array.
 * @param partial_sums The partial sums array.
 * @param blockSize The size of the blocks.
 * @param len The length of the arrays.
 */
template <typename T>
__global__ void scan_add(T* input, T* partial_sums, size_t blockSize, size_t len) {
   const T val = partial_sums[blockIdx.x];
   const size_t target1 = 2 * blockIdx.x * blockDim.x + threadIdx.x;
   const size_t target2 = target1 + blockDim.x;
   if (target1 < len) {
      input[target1] += val;
      if (target2 < len) {
         input[target2] += val;
      }
   }
}

/**
 * @brief GPU kernel for performing scan-reduce operation.
 *
 * This kernel performs a scan-reduce operation on the given input SplitVector, applying the specified rule.
 * It computes the total number of valid elements using the rule and stores the results in the output SplitVector.
 * Part of splitvector's stream compaction mechanism.
 *
 * @tparam T Type of the array elements.
 * @tparam Rule The rule functor for element validation.
 * @param input The input SplitVector.
 * @param output The output SplitVector for storing the results.
 * @param rule The rule functor object.
 */
template <typename T, typename Rule>
__global__ void scan_reduce(split::SplitVector<T, split::split_unified_allocator<T>>* input,
                            split::SplitVector<uint32_t, split::split_unified_allocator<uint32_t>>* output, Rule rule) {

   size_t size = input->size();
   size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < size) {
      size_t total_valid_elements = __syncthreads_count(rule(input->at(tid)));
      if (threadIdx.x == 0) {
         output->at(blockIdx.x) = total_valid_elements;
      }
   }
}

/**
 * @brief Prefix Scan routine with Bank Conflict Optimization.
 *
 * This function performs a prefix scan operation on the given input array, using the bank conflict optimization.
 * It utilizes shared memory.
 * Part of splitvector's stream compaction mechanism.
 * Credits to  https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
 *
 * @tparam T Type of the array elements.
 * @param input The input array.
 * @param output The output array for storing the prefix scan results.
 * @param partial_sums The array of partial sums for each block.
 * @param n The number of elements per block.
 * @param len The length of the arrays.
 */
template <typename T>
__global__ void split_prescan(T* input, T* output, T* partial_sums, int n, size_t len) {

   extern __shared__ T buffer[];
   int tid = threadIdx.x;
   int offset = 1;
   int local_start = blockIdx.x * n;
   input = input + local_start;
   output = output + local_start;

   // Load into shared memory
   int ai = tid;
   int bi = tid + (n / 2);
   int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
   int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

   if (local_start + ai < len && local_start + bi < len) {
      buffer[ai + bankOffsetA] = input[ai];
      buffer[bi + bankOffsetB] = input[bi];
   }

   // Reduce Phase
   for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();

      if (tid < d) {
         int ai = offset * (2 * tid + 1) - 1;
         int bi = offset * (2 * tid + 2) - 1;
         ai += CONFLICT_FREE_OFFSET(ai);
         bi += CONFLICT_FREE_OFFSET(bi);

         buffer[bi] += buffer[ai];
      }
      offset *= 2;
   }

   // Exclusive scan so zero out last element (will propagate to first)
   if (tid == 0) {
      partial_sums[blockIdx.x] = buffer[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
      buffer[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
   }

   // Downsweep Phase
   for (int d = 1; d < n; d *= 2) {

      offset >>= 1;
      __syncthreads();
      if (tid < d) {
         int ai = offset * (2 * tid + 1) - 1;
         int bi = offset * (2 * tid + 2) - 1;
         ai += CONFLICT_FREE_OFFSET(ai);
         bi += CONFLICT_FREE_OFFSET(bi);

         T tmp = buffer[ai];
         buffer[ai] = buffer[bi];
         buffer[bi] += tmp;
      }
   }
   __syncthreads();
   if (local_start + ai < len && local_start + bi < len) {
      output[ai] = buffer[ai + bankOffsetA];
      output[bi] = buffer[bi + bankOffsetB];
   }
}

/**
 * @brief Kernel for compacting elements based on a rule.
 *
 * This kernel performs element compaction on the given input SplitVector based on a specified rule.
 * It generates compacted output and updates counts and offsets SplitVectors.
 * Part of splitvector's stream compaction mechanism.
 *
 * @tparam T Type of the array elements.
 * @tparam Rule The rule functor for element compaction.
 * @tparam BLOCKSIZE The size of each thread block.
 * @tparam WARP The size of each warp.
 * @param input The input SplitVector.
 * @param counts The SplitVector for storing counts of valid elements.
 * @param offsets The SplitVector for storing offsets of valid elements.
 * @param output The output SplitVector for storing the compacted elements.
 * @param rule The rule functor object.
 */
template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
__global__ void split_compact(split::SplitVector<T, split::split_unified_allocator<T>>* input,
                              split::SplitVector<uint32_t, split::split_unified_allocator<uint32_t>>* counts,
                              split::SplitVector<uint32_t, split::split_unified_allocator<uint32_t>>* offsets,
                              split::SplitVector<T, split::split_unified_allocator<T>>* output, Rule rule) {
   extern __shared__ uint32_t buffer[];
   const size_t size = input->size();
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid >= size) {
      return;
   }
   unsigned int offset = BLOCKSIZE / WARP;
   const unsigned int wid = tid / WARP;
   const unsigned int widb = threadIdx.x / WARP;
   const unsigned int w_tid = tid % WARP;
   const unsigned int warps_in_block = blockDim.x / WARP;
   const bool tres = rule(input->at(tid));

   auto mask = split::s_warpVote(tres, SPLIT_VOTING_MASK);
#ifdef __NVCC__
   uint32_t n_neighbors = mask & ((1 << w_tid) - 1);
#else
   uint64_t n_neighbors = mask & ((1ul << w_tid) - 1);
#endif
   auto total_valid_in_warp = split::s_pop_count(mask);
   if (w_tid == 0) {
      buffer[widb] = total_valid_in_warp;
   }
   __syncthreads();
   if (w_tid == 0 && wid % warps_in_block == 0) {
      buffer[offset + widb] = 0;
      for (unsigned int i = 0; i < warps_in_block - 1; ++i) {
         buffer[offset + widb + i + 1] = buffer[offset + widb + i] + buffer[widb + i];
      }
   }
   __syncthreads();
   const unsigned int neighbor_count = split::s_pop_count(n_neighbors);
   const unsigned int private_index = buffer[offset + widb] + offsets->at(wid / warps_in_block) + neighbor_count;
   if (tres && widb != warps_in_block) {
      output->at(private_index) = input->at(tid);
   }
   __syncthreads();
   if (tid == 0) {
      const unsigned int actual_total_blocks = offsets->back() + counts->back();
      output->erase(&output->at(actual_total_blocks), output->end());
   }
}

/**
 * @brief Same as split_compact but only for hashinator keys.
 */
template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
__global__ void split_compact_keys(split::SplitVector<T, split::split_unified_allocator<T>>* input,
                                   split::SplitVector<uint32_t, split::split_unified_allocator<uint32_t>>* counts,
                                   split::SplitVector<uint32_t, split::split_unified_allocator<uint32_t>>* offsets,
                                   split::SplitVector<U, split::split_unified_allocator<U>>* output, Rule rule) {
   extern __shared__ uint32_t buffer[];
   const size_t size = input->size();
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid >= size) {
      return;
   }
   unsigned int offset = BLOCKSIZE / WARP;
   const unsigned int wid = tid / WARP;
   const unsigned int widb = threadIdx.x / WARP;
   const unsigned int w_tid = tid % WARP;
   const unsigned int warps_in_block = blockDim.x / WARP;
   const bool tres = rule(input->at(tid));

   auto mask = split::s_warpVote(tres, SPLIT_VOTING_MASK);
#ifdef __NVCC__
   uint32_t n_neighbors = mask & ((1 << w_tid) - 1);
#else
   uint64_t n_neighbors = mask & ((1ul << w_tid) - 1);
#endif
   auto total_valid_in_warp = split::s_pop_count(mask);
   if (w_tid == 0) {
      buffer[widb] = total_valid_in_warp;
   }
   __syncthreads();
   if (w_tid == 0 && wid % warps_in_block == 0) {
      buffer[offset + widb] = 0;
      for (unsigned int i = 0; i < warps_in_block - 1; ++i) {
         buffer[offset + widb + i + 1] = buffer[offset + widb + i] + buffer[widb + i];
      }
   }
   __syncthreads();
   const unsigned int neighbor_count = split::s_pop_count(n_neighbors);
   const unsigned int private_index = buffer[offset + widb] + offsets->at(wid / warps_in_block) + neighbor_count;
   if (tres && widb != warps_in_block) {
      output->at(private_index) = (input->at(tid)).first;
   }

   __syncthreads();
   if (tid == 0) {
      const unsigned int actual_total_blocks = offsets->back() + counts->back();
      output->erase(&output->at(actual_total_blocks), output->end());
   }
}

/**
 * @brief Same as split_compact_keys but uses raw memory
 */
template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
__global__ void split_compact_keys_raw(T* input, uint32_t* counts, uint32_t* offsets, U* output, Rule rule,
                                       const size_t size, size_t nBlocks, uint32_t* retval) {

   extern __shared__ uint32_t buffer[];
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid >= size) {
      return;
   }
   unsigned int offset = BLOCKSIZE / WARP;
   const unsigned int wid = tid / WARP;
   const unsigned int widb = threadIdx.x / WARP;
   const unsigned int w_tid = tid % WARP;
   const unsigned int warps_in_block = blockDim.x / WARP;
   const bool tres = rule(input[tid]);

   auto mask = split::s_warpVote(tres, SPLIT_VOTING_MASK);
#ifdef __NVCC__
   uint32_t n_neighbors = mask & ((1 << w_tid) - 1);
#else
   uint64_t n_neighbors = mask & ((1ul << w_tid) - 1);
#endif
   auto total_valid_in_warp = split::s_pop_count(mask);
   if (w_tid == 0) {
      buffer[widb] = total_valid_in_warp;
   }
   __syncthreads();
   if (w_tid == 0 && wid % warps_in_block == 0) {
      buffer[offset + widb] = 0;
      for (unsigned int i = 0; i < warps_in_block - 1; ++i) {
         buffer[offset + widb + i + 1] = buffer[offset + widb + i] + buffer[widb + i];
      }
   }
   __syncthreads();
   const unsigned int neighbor_count = split::s_pop_count(n_neighbors);
   const unsigned int private_index = buffer[offset + widb] + offsets[(wid / warps_in_block)] + neighbor_count;
   if (tres && widb != warps_in_block) {
      output[private_index] = input[tid].first;
   }
   if (tid == 0) {
      // const unsigned int actual_total_blocks=offsets->back()+counts->back();
      *retval = offsets[nBlocks - 1] + counts[nBlocks - 1];
   }
}

/**
 * @brief Perform prefix scan operation with Bank Conflict Optimization.
 *
 * This function performs a prefix scan operation on the given SplitVector, utilizing bank conflict optimization.
 * It utilizes shared memory.
 *
 * @tparam T Type of the array elements.
 * @tparam BLOCKSIZE The size of each thread block.
 * @tparam WARP The size of each warp.
 * @param input The input SplitVector.
 * @param output The output SplitVector for storing the prefix scan results.
 * @param s The split_gpuStream_t stream for GPU execution (default is 0).
 */
template <typename T, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void split_prefix_scan(split::SplitVector<T, split::split_unified_allocator<T>>& input,
                       split::SplitVector<T, split::split_unified_allocator<T>>& output, split_gpuStream_t s = 0)

{
   using vector = split::SplitVector<T, split::split_unified_allocator<T>>;

   // Input size
   const size_t input_size = input.size();

   // Scan is performed in half Blocksizes
   size_t scanBlocksize = BLOCKSIZE / 2;
   size_t scanElements = 2 * scanBlocksize;
   size_t gridSize = input_size / scanElements;

   // If input is not exactly divisible by scanElements we launch an extra block
   assert(isPow2(input_size) && "Using prefix scan with non powers of 2 as input size is not thought out yet :D");

   if (input_size % scanElements != 0) {
      gridSize = 1 << ((int)ceil(log(++gridSize) / log(2)));
   }

   // If the elements are too few manually override and launch a small kernel
   if (input_size < scanElements) {
      scanBlocksize = input_size / 2;
      scanElements = input_size;
      if (scanBlocksize == 0) {
         scanBlocksize += 1;
      }
      gridSize = 1;
   }

   // Allocate memory for partial sums
   vector partial_sums(gridSize);

   // TODO +FIXME extra shmem allocations
   split::tools::split_prescan<<<gridSize, scanBlocksize, 2 * scanElements * sizeof(T), s>>>(
       input.data(), output.data(), partial_sums.data(), scanElements, input.size());
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

   if (gridSize > 1) {
      if (partial_sums.size() <= scanElements) {
         vector partial_sums_dummy(gridSize);
         // TODO +FIXME extra shmem allocations
         split::tools::split_prescan<<<1, scanBlocksize, 2 * scanElements * sizeof(T), s>>>(
             partial_sums.data(), partial_sums.data(), partial_sums_dummy.data(), gridSize, partial_sums.size());
         SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
      } else {
         vector partial_sums_clone(partial_sums);
         split_prefix_scan<T, BLOCKSIZE, WARP>(partial_sums_clone, partial_sums, s);
      }
      split::tools::scan_add<<<gridSize, scanBlocksize, 0, s>>>(output.data(), partial_sums.data(), scanElements,
                                                                output.size());
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }
}

/**
 * @brief Computes the next power of 2 greater than or equal to a given value.
 *
 * @param v The value for which to compute the next power of 2.
 * @return The next power of 2 greater than or equal to the input value.
 * Modified from (http://www-graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2) to support 64-bit uints
 * Included here as well for standalone use of splitvec outside of hashintor
 */
constexpr inline size_t nextPow2(size_t v) noexcept {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}

/**
 * @brief Same as split_compact but with raw memory
 */
template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
__global__ void split_compact_raw(T* input, uint32_t* counts, uint32_t* offsets, T* output, Rule rule,
                                  const size_t size, size_t nBlocks, uint32_t* retval) {

   extern __shared__ uint32_t buffer[];
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid >= size) {
      return;
   }
   unsigned int offset = BLOCKSIZE / WARP;
   const unsigned int wid = tid / WARP;
   const unsigned int widb = threadIdx.x / WARP;
   const unsigned int w_tid = tid % WARP;
   const unsigned int warps_in_block = blockDim.x / WARP;
   const bool tres = rule(input[tid]);

   auto mask = split::s_warpVote(tres, SPLIT_VOTING_MASK);
#ifdef __NVCC__
   uint32_t n_neighbors = mask & ((1 << w_tid) - 1);
#else
   uint64_t n_neighbors = mask & ((1ul << w_tid) - 1);
#endif
   auto total_valid_in_warp = split::s_pop_count(mask);
   if (w_tid == 0) {
      buffer[widb] = total_valid_in_warp;
   }
   __syncthreads();
   if (w_tid == 0 && wid % warps_in_block == 0) {
      buffer[offset + widb] = 0;
      for (unsigned int i = 0; i < warps_in_block - 1; ++i) {
         buffer[offset + widb + i + 1] = buffer[offset + widb + i] + buffer[widb + i];
      }
   }
   __syncthreads();
   const unsigned int neighbor_count = split::s_pop_count(n_neighbors);
   const unsigned int private_index = buffer[offset + widb] + offsets[(wid / warps_in_block)] + neighbor_count;
   if (tres && widb != warps_in_block) {
      output[private_index] = input[tid];
   }
   if (tid == 0) {
      // const unsigned int actual_total_blocks=offsets->back()+counts->back();
      *retval = offsets[nBlocks - 1] + counts[nBlocks - 1];
   }
}

/**
 * @brief CUDA Memory Pool class for managing GPU memory.
 *
 * This class provides a simle memory pool implementation for allocating and deallocating GPU memory.
 * It uses async mallocs
 *
 */
class Cuda_mempool {
private:
   size_t total_bytes;
   size_t bytes_used;
   void* _data;
   split_gpuStream_t s;
   bool isOwner;

public:
   explicit Cuda_mempool(size_t bytes, split_gpuStream_t str) {
      s = str;
      SPLIT_CHECK_ERR(split_gpuMallocAsync(&_data, bytes, s));
      total_bytes = bytes;
      bytes_used = 0;
      isOwner = true;
   }
   explicit Cuda_mempool(void* ptr, size_t bytes) {
      total_bytes = bytes;
      bytes_used = 0;
      isOwner = false;
      _data = ptr;
   }

   Cuda_mempool() = delete;
   Cuda_mempool(const Cuda_mempool& other) = delete;
   Cuda_mempool(Cuda_mempool&& other) = delete;
   ~Cuda_mempool() {
      if (isOwner) {
         SPLIT_CHECK_ERR(split_gpuFreeAsync(_data, s));
      }
   }

   void* allocate(const size_t bytes) {
      assert(bytes_used + bytes < total_bytes && "Mempool run out of space and crashed!");
      void* ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(_data) + bytes_used);
      bytes_used += bytes;
      return ptr;
   };

   void deallocate(const size_t bytes) { bytes_used -= bytes; };
   void reset() { bytes_used = 0; }
   const size_t& fill() const { return bytes_used; }
   const size_t& capacity() const { return total_bytes; }
   size_t free_space() const { return total_bytes - bytes_used; }
};

/**
 * @brief Same as scan_reduce but with raw memory
 */
template <typename T, typename Rule>
__global__ void scan_reduce_raw(T* input, uint32_t* output, Rule rule, size_t size) {

   size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < size) {
      size_t total_valid_elements = __syncthreads_count(rule(input[tid]));
      if (threadIdx.x == 0) {
         output[blockIdx.x] = total_valid_elements;
      }
   }
}

/**
 * @brief Same as split_prefix_scan but with raw memory
 */
template <typename T, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void split_prefix_scan_raw(T* input, T* output, Cuda_mempool& mPool, const size_t input_size, split_gpuStream_t s = 0) {

   // Scan is performed in half Blocksizes
   size_t scanBlocksize = BLOCKSIZE / 2;
   size_t scanElements = 2 * scanBlocksize;
   size_t gridSize = input_size / scanElements;

   // If input is not exactly divisible by scanElements we launch an extra block
   assert(isPow2(input_size) && "Using prefix scan with non powers of 2 as input size is not thought out yet :D");
   if (input_size % scanElements != 0) {
      gridSize = 1 << ((int)ceil(log(++gridSize) / log(2)));
   }
   // If the elements are too few manually override and launch a small kernel
   if (input_size < scanElements) {
      scanBlocksize = input_size / 2;
      scanElements = input_size;
      if (scanBlocksize == 0) {
         scanBlocksize += 1;
      }
      gridSize = 1;
   }

   // Allocate memory for partial sums
   T* partial_sums = (T*)mPool.allocate(gridSize * sizeof(T));
   // TODO + FIXME extra shmem
   split::tools::split_prescan<<<gridSize, scanBlocksize, 2 * scanElements * sizeof(T), s>>>(
       input, output, partial_sums, scanElements, input_size);
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

   if (gridSize > 1) {
      if (gridSize <= scanElements) {
         T* partial_sums_dummy = (T*)mPool.allocate(sizeof(T));
         // TODO + FIXME extra shmem
         split::tools::split_prescan<<<1, scanBlocksize, 2 * scanElements * sizeof(T), s>>>(
             partial_sums, partial_sums, partial_sums_dummy, gridSize, gridSize * sizeof(T));
         SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
      } else {
         T* partial_sums_clone = (T*)mPool.allocate(gridSize * sizeof(T));
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(partial_sums_clone, partial_sums, gridSize * sizeof(T),
                                              split_gpuMemcpyDeviceToDevice, s));
         split_prefix_scan_raw(partial_sums_clone, partial_sums, mPool, gridSize, s);
      }
      split::tools::scan_add<<<gridSize, scanBlocksize, 0, s>>>(output, partial_sums, scanElements, input_size);
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   }
}

/**
 * @brief Same as copy_if but using raw memory
 */
template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
uint32_t copy_if_raw(T* input, T* output, size_t inputSize, Rule rule, size_t nBlocks, Cuda_mempool& mPool,
                     split_gpuStream_t s = 0) {

   uint32_t* d_counts;
   uint32_t* d_offsets;
   d_counts = (uint32_t*)mPool.allocate(nBlocks * sizeof(uint32_t));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   SPLIT_CHECK_ERR(split_gpuMemsetAsync(d_counts, 0, nBlocks * sizeof(uint32_t),s));

   // Phase 1 -- Calculate per warp workload
   split::tools::scan_reduce_raw<<<nBlocks, BLOCKSIZE, 0, s>>>(input, d_counts, rule, inputSize);
   d_offsets = (uint32_t*)mPool.allocate(nBlocks * sizeof(uint32_t));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   SPLIT_CHECK_ERR(split_gpuMemsetAsync(d_offsets, 0, nBlocks * sizeof(uint32_t),s));

   // Step 2 -- Exclusive Prefix Scan on offsets
   if (nBlocks == 1) {
      split_prefix_scan_raw<uint32_t, 2, WARP>(d_counts, d_offsets, mPool, nBlocks, s);
   } else {
      split_prefix_scan_raw<uint32_t, BLOCKSIZE, WARP>(d_counts, d_offsets, mPool, nBlocks, s);
   }
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

   // Step 3 -- Compaction
   uint32_t* retval = (uint32_t*)mPool.allocate(sizeof(uint32_t));
   split::tools::split_compact_raw<T, Rule, BLOCKSIZE, WARP>
       <<<nBlocks, BLOCKSIZE, 2 * (BLOCKSIZE / WARP) * sizeof(unsigned int), s>>>(input, d_counts, d_offsets, output,
                                                                                  rule, inputSize, nBlocks, retval);
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   uint32_t numel;
   SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&numel, retval, sizeof(uint32_t), split_gpuMemcpyDeviceToHost, s));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   return numel;
}

/**
 * @brief Same as copy_keys_if but using raw memory
 */
template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
size_t copy_keys_if_raw(T* input, U* output, size_t inputSize, Rule rule, size_t nBlocks, Cuda_mempool& mPool,
                        split_gpuStream_t s = 0) {

   uint32_t* d_counts;
   uint32_t* d_offsets;
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   d_counts = (uint32_t*)mPool.allocate(nBlocks * sizeof(uint32_t));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   SPLIT_CHECK_ERR(split_gpuMemsetAsync(d_counts, 0, nBlocks * sizeof(uint32_t),s));

   // Phase 1 -- Calculate per warp workload
   split::tools::scan_reduce_raw<<<nBlocks, BLOCKSIZE, 0, s>>>(input, d_counts, rule, inputSize);
   d_offsets = (uint32_t*)mPool.allocate(nBlocks * sizeof(uint32_t));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   SPLIT_CHECK_ERR(split_gpuMemsetAsync(d_offsets, 0, nBlocks * sizeof(uint32_t),s));

   // Step 2 -- Exclusive Prefix Scan on offsets
   if (nBlocks == 1) {
      split_prefix_scan_raw<uint32_t, 2, WARP>(d_counts, d_offsets, mPool, nBlocks, s);
   } else {
      split_prefix_scan_raw<uint32_t, BLOCKSIZE, WARP>(d_counts, d_offsets, mPool, nBlocks, s);
   }
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));

   // Step 3 -- Compaction
   uint32_t* retval = (uint32_t*)mPool.allocate(sizeof(uint32_t));
   split::tools::split_compact_keys_raw<T, U, Rule, BLOCKSIZE, WARP>
       <<<nBlocks, BLOCKSIZE, 2 * (BLOCKSIZE / WARP) * sizeof(unsigned int), s>>>(input, d_counts, d_offsets, output,
                                                                                  rule, inputSize, nBlocks, retval);
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   uint32_t numel;
   SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&numel, retval, sizeof(uint32_t), split_gpuMemcpyDeviceToHost, s));
   SPLIT_CHECK_ERR(split_gpuStreamSynchronize(s));
   return numel;
}

/**
 * @brief Estimates memory needed for compacting the input splitvector
 */
template <typename T, int BLOCKSIZE = 1024>
[[nodiscard]] size_t
estimateMemoryForCompaction(const split::SplitVector<T, split::split_unified_allocator<T>>& input) noexcept {
   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }

   // Allocate with Mempool
   return 8 * nBlocks * sizeof(uint32_t);
}

/**
 * @brief Estimates memory needed for compacting the input splitvector
 */
template <int BLOCKSIZE = 1024>
[[nodiscard]] size_t estimateMemoryForCompaction(const size_t inputSize) noexcept {
   // Figure out Blocks to use
   size_t _s = std::ceil((float(inputSize)) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }

   // Allocate with Mempool
   return 8 * nBlocks * sizeof(uint32_t);
}

/**
 * @brief Same as copy_if but only for Hashinator keys
 */
template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_keys_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
                  split::SplitVector<U, split::split_unified_allocator<U>>& output, Rule rule,
                  split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }

   // Allocate with Mempool
   const size_t memory_for_pool = 8 * nBlocks * sizeof(uint32_t);
   Cuda_mempool mPool(memory_for_pool, s);
   auto len = copy_keys_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(&output[len], output.end());
}

/**
 * @brief Perform element compaction based on a rule.
 *
 * This function performs element compaction on the given input SplitVector based on a specified rule.
 * It generates compacted output, updates counts and offsets SplitVectors, and returns compacted count.
 *
 * @tparam T Type of the array elements.
 * @tparam Rule The rule functor for element compaction.
 * @tparam BLOCKSIZE The size of each thread block.
 * @tparam WARP The size of each warp.
 * @param input The input SplitVector.
 * @param output The output SplitVector for storing the compacted elements.
 * @param rule The rule functor object.
 * @param s The split_gpuStream_t stream for GPU execution (default is 0).
 */
template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
             split::SplitVector<T, split::split_unified_allocator<T>>& output, Rule rule, split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }

   // Allocate with Mempool
   const size_t memory_for_pool = 8 * nBlocks * sizeof(uint32_t);
   Cuda_mempool mPool(memory_for_pool, s);
   auto len = copy_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(&output[len], output.end());
}

template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_keys_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
                  split::SplitVector<U, split::split_unified_allocator<U>>& output, Rule rule, Cuda_mempool&& mPool,
                  split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   auto len =
       copy_keys_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, std::forward<Cuda_mempool>(mPool), s);
   output.erase(&output[len], output.end());
}

template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
             split::SplitVector<T, split::split_unified_allocator<T>>& output, Rule rule, Cuda_mempool&& mPool,
             split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   auto len = copy_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(&output[len], output.end());
}

template <typename T, typename U, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_keys_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
                  split::SplitVector<U, split::split_unified_allocator<U>>& output, Rule rule, void* stack,
                  size_t max_size, split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   assert(stack && "Invalid stack!");
   Cuda_mempool mPool(stack, max_size);
   auto len = copy_keys_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(&output[len], output.end());
}

template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_if(split::SplitVector<T, split::split_unified_allocator<T>>& input,
             split::SplitVector<T, split::split_unified_allocator<T>>& output, Rule rule, void* stack, size_t max_size,
             split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   assert(stack && "Invalid stack!");
   Cuda_mempool mPool(stack, max_size);
   auto len = copy_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(&output[len], output.end());
}

#ifndef SPLIT_CPU_ONLY_MODE
template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
void copy_if(split::DeviceVector<T>& input,
                split::DeviceVector<T>& output, Rule rule, void* stack, size_t max_size,
                split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(input.size())) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   assert(stack && "Invalid stack!");
   Cuda_mempool mPool(stack, max_size);
   auto len = copy_if_raw(input.data(), output.data(), input.size(), rule, nBlocks, mPool, s);
   output.erase(output.data()+len, output.end());
}
#endif

template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
[[nodiscard]] size_t copy_if(T* input, T* output, size_t inputSize, Rule rule, Cuda_mempool& mPool,
                             split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(inputSize)) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   return copy_if_raw(input, output, inputSize, rule, nBlocks, mPool, s);
}

template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
[[nodiscard]] size_t copy_if(T* input, T* output, size_t inputSize, Rule rule, void* stack, size_t max_size,split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(inputSize)) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   assert(stack && "Invalid stack!");
   Cuda_mempool mPool(stack, max_size);
   return copy_if_raw(input, output, inputSize, rule, nBlocks, mPool, s);
}

template <typename T, typename Rule, size_t BLOCKSIZE = 1024, size_t WARP = WARPLENGTH>
[[nodiscard]] size_t copy_if(T* input, T* output, size_t inputSize, Rule rule, split_gpuStream_t s = 0) {

   // Figure out Blocks to use
   size_t _s = std::ceil((float(inputSize)) / (float)BLOCKSIZE);
   size_t nBlocks = nextPow2(_s);
   if (nBlocks == 0) {
      nBlocks += 1;
   }
   size_t mem = estimateMemoryForCompaction(inputSize);
   Cuda_mempool mPool(mem, s);
   return copy_if_raw(input, output, inputSize, rule, nBlocks, mPool, s);
}

} // namespace tools
} // namespace split

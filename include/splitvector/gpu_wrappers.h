/* File:    gpu_wrappers.h
 * Authors: Kostis Papadakis (2023)
 *
 * This file defines wrappers over GPU intrinsics for NVIDIA and
 * AMD hardware
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
#ifndef SPLIT_CPU_ONLY_MODE

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#elif __HIP__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif
namespace split {

/**
 * @brief Wrapper for atomic exchange operation.
 *
 * @tparam T The data type of the value being exchanged.
 * @param address Pointer to the memory location.
 * @param val The value to exchange.
 * @return The value that was replaced.
 */
template <typename T>
__device__ __forceinline__ T s_atomicExch(T* address, T val) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
   if constexpr (sizeof(T) == 4) {
      return atomicExch((unsigned int*)address, (unsigned int)val);
   } else if constexpr (sizeof(T) == 8) {
      return atomicExch((unsigned long long*)address, (unsigned long long)val);
   } else {
      // Cannot be static_assert(false...);
      static_assert(!sizeof(T*), "Not supported");
   }
}

/**
 * @brief Wrapper for atomic compare-and-swap operation.
 *
 * @tparam T The data type of the value being compared and swapped.
 * @param address Pointer to the memory location.
 * @param compare Predicate.
 * @param val The value to swap.
 * @return The original value at the memory location.
 */
template <typename T>
__device__ __forceinline__ T s_atomicCAS(T* address, T compare, T val) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
   if constexpr (sizeof(T) == 4) {
      return atomicCAS((unsigned int*)address, (unsigned int)compare, (unsigned int)val);
   } else if constexpr (sizeof(T) == 8) {
      return atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
   } else {
      // Cannot be static_assert(false...);
      static_assert(!sizeof(T*), "Not supported");
   }
}

/**
 * @brief Wrapper for atomic addition operation.
 *
 * @tparam T The data type of the value being added.
 * @tparam U The data type of the value to add.
 * @param address Pointer to the memory location.
 * @param val The value to add.
 * @return The original value at the memory location.
 */
template <typename T, typename U>
__device__ __forceinline__ T s_atomicAdd(T* address, U val) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
   if constexpr (sizeof(T) == 4) {
      if constexpr (std::is_signed<T>::value) {
         return atomicAdd((int*)address, static_cast<T>(val));
      } else {
         return atomicAdd((unsigned int*)address, static_cast<T>(val));
      }
   } else if constexpr (sizeof(T) == 8) {
      return atomicAdd((unsigned long long*)address, static_cast<T>(val));
   } else {
      // Cannot be static_assert(false...);
      static_assert(!sizeof(T*), "Not supported");
   }
}

/**
 * @brief Wrapper for atomic subtraction operation.
 *
 * @tparam T The data type of the value being subtracted.
 * @tparam U The data type of the value to subtract.
 * @param address Pointer to the memory location.
 * @param val The value to subtract.
 * @return The original value at the memory location.
 */
template <typename T, typename U>
__device__ __forceinline__ T s_atomicSub(T* address, U val) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
   if constexpr (sizeof(T) == 4) {
      if constexpr (std::is_signed<T>::value) {
         return atomicSub((int*)address, static_cast<T>(val));
      } else {
         return atomicSub((unsigned int*)address, static_cast<T>(val));
      }
   } else {
      // Cannot be static_assert(false...);
      static_assert(!sizeof(T*), "Not supported");
   }
}

/**
 * @brief Returns the submask needed by each Virtual Warp during voting (CUDA variant).
 */
[[nodiscard]] __device__ __forceinline__ uint32_t constexpr getIntraWarpMask_CUDA(uint32_t n, uint32_t l,
                                                                                  uint32_t r) noexcept {
   uint32_t num = ((1 << r) - 1) ^ ((1 << (l - 1)) - 1);
   return (n ^ num);
};

/**
 * @brief Returns the submask needed by each Virtual Warp during voting (AMD variant).
 */
[[nodiscard]] __device__ __forceinline__ uint64_t constexpr getIntraWarpMask_AMD(uint64_t n, uint64_t l,
                                                                                 uint64_t r) noexcept {
   uint64_t num = ((1ull << r) - 1) ^ ((1ull << (l - 1)) - 1);
   return (n ^ num);
};

/**
 * @brief Wrapper for warp-level voting (ballot) operation.
 *
 * @tparam T The data type of the voting mask.
 * @param predicate The predicate value.
 * @param votingMask The voting mask.
 * @return Result mask.
 */
template <typename T>
__device__ __forceinline__ T s_warpVote(bool predicate, T votingMask = T(-1)) noexcept {
#ifdef __NVCC__
   return __ballot_sync(votingMask, predicate);
#endif

#ifdef __HIP__
   return __ballot(predicate);
#endif
}

/**
 * @brief Wrapper for finding the index of the first set bit in a mask.
 *
 * @tparam T The data type of the mask.
 * @param mask The mask value.
 * @return The index of the first set bit.
 */
template <typename T>
__device__ __forceinline__ int s_findFirstSig(T mask) noexcept {
#ifdef __NVCC__
   return __ffs(mask);
#endif

#ifdef __HIP__
   return __ffsll((unsigned long long)mask);
#endif
}

/**
 * @brief Wrapper for warp-level voting (any) operation.
 *
 * @tparam T The data type of the voting mask.
 * @param predicate The predicate value.
 * @param votingMask The voting mask.
 * @return Whether any of the warp-threads satisfy the predicate.
 */
template <typename T>
__device__ __forceinline__ int s_warpVoteAny(bool predicate, T votingMask = T(-1)) noexcept {
#ifdef __NVCC__
   return __any_sync(votingMask, predicate);
#endif

#ifdef __HIP__
   return __any(predicate);
#endif
}

/**
 * @brief Wrapper for counting the number of set bits in a mask.
 *
 * @tparam T The data type of the mask.
 * @param mask The mask value.
 * @return The number of set bits.
 */
template <typename T>
__device__ __forceinline__ uint32_t s_pop_count(T mask) noexcept {
#ifdef __NVCC__
   return __popc(mask);
#endif
#ifdef __HIP__
   if constexpr (sizeof(T) == 4) {
      return __popc(mask);
   } else if constexpr (sizeof(mask) == 8) {
      return __popcll(mask);
   } else {
      // Cannot be static_assert(false...);
      static_assert(!sizeof(T*), "Not supported");
   }
#endif
}

/**
 * @brief Wrapper for performing a broadcast shuffle operation.
 *
 * @tparam T The data type of the variable.
 * @tparam U The data type of the mask.
 * @param variable The variable to shuffle.
 * @param  source the lane source to broadcast from.
 * @param mask Voting mask.
 * @return The shuffled variable.
 */
template <typename T, typename U>
__device__ __forceinline__ T s_shuffle(T variable, unsigned int source, U mask = 0) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
#ifdef __NVCC__
   return __shfl_sync(mask, variable, source);
#endif
#ifdef __HIP__
   return __shfl(variable, source);
#endif
}

/**
 * @brief Wrapper for performing a down register shuffle operation.
 *
 * @tparam T The data type of the variable.
 * @tparam U The data type of the mask.
 * @param variable The variable to shuffle.
 * @param delta The offset.
 * @param mask Voting mask.
 * @return The shuffled variable.
 */
template <typename T, typename U>
__device__ __forceinline__ T s_shuffle_down(T variable, unsigned int delta, U mask = 0) noexcept {
   static_assert(std::is_integral<T>::value && "Only integers supported");
#ifdef __NVCC__
   return __shfl_down_sync(mask, variable, delta);
#endif
#ifdef __HIP__
   return __shfl_down(variable, delta);
#endif
}
} // namespace split
#endif

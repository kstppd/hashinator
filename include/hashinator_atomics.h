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
#ifndef HASHINATOR_HOST_ONLY
#include <cuda_runtime.h> 
#include "common.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


namespace Hashinator{



   template <typename T>
   HASHINATOR_DEVICEONLY
   __forceinline__
   T h_atomicExch(T* address, T val){
      static_assert(std::is_integral<T>::value && "Only integers supported");
      if constexpr(sizeof(T)==4){
         return atomicExch((unsigned int*)address,(unsigned int)val);
      }else if constexpr(sizeof(T)==8){
         return atomicExch((unsigned long long*)address,(unsigned long long)val);
      }else{
         //Cannot be static_assert(false...);
         static_assert(!sizeof(T*), "Not supported");
      }
   }


   template <typename T>
   HASHINATOR_DEVICEONLY
   __forceinline__
   T h_atomicCAS(T* address,T compare, T val){
      static_assert(std::is_integral<T>::value && "Only integers supported");
      if constexpr(sizeof(T)==4){
         return atomicCAS((unsigned int*)address,(unsigned int)compare,(unsigned int)val);
      }else if constexpr(sizeof(T)==8){
         return atomicCAS((unsigned long long*)address,(unsigned long long)compare,(unsigned long long)val);
      }else{
         //Cannot be static_assert(false...);
         static_assert(!sizeof(T*), "Not supported");
      }
   }


   template <typename T,typename U>
   HASHINATOR_DEVICEONLY
   __forceinline__
   T h_atomicAdd(T* address, U val){
      static_assert(std::is_integral<T>::value && "Only integers supported");
      if constexpr(sizeof(T)==4){
         if constexpr(std::is_signed<T>::value){
            return  atomicAdd((int*)address,static_cast<T>(val));
         }else{
            return  atomicAdd((unsigned int*)address,static_cast<T>(val));
         }
      }else if constexpr(sizeof(T)==8){
            return  atomicAdd((unsigned long long*)address,static_cast<T>(val));
      }else{
         //Cannot be static_assert(false...);
         static_assert(!sizeof(T*), "Not supported");
      }
   }
   

   template <typename T,typename U>
   HASHINATOR_DEVICEONLY
   __forceinline__
   T h_atomicSub(T* address, U val){
      static_assert(std::is_integral<T>::value && "Only integers supported");
      if constexpr(sizeof(T)==4){
         if constexpr(std::is_signed<T>::value){
            return  atomicSub((int*)address,static_cast<T>(val));
         }else{
            return  atomicSub((unsigned int*)address,static_cast<T>(val));
         }
      }else{
         //Cannot be static_assert(false...);
         static_assert(!sizeof(T*), "Not supported");
      }
   }



   /*
    * Returns the submask needed by each Virtual Warp during voting
    */
   template <typename T>
   __device__  __forceinline__
   T getIntraWarpMask(T n ,T l ,T r){
      uint32_t num = ((T(1)<<r)-1)^((1<<(T(1)-1))-1);
      return (n^num);
   };
   
   /*
    * Wraps over ballots for AMD and NVIDIA
    */
   template <typename T>
   __device__  __forceinline__
   T warpVote(bool predicate,T votingMask=T(-1)){
      #ifdef __NVCC__
      return __ballot_sync(votingMask,predicate);
      #endif 

      #ifdef __HIP_PLATFORM_HCC___
      return __ballot(predicate);
      #endif 
   }

   /*
    * Wraps over __ffs for AMD and NVIDIA
    */
   template <typename T>
   __device__  __forceinline__
   int findFirstSig(T mask){
      #ifdef __NVCC__
      return __ffs ( mask);
      #endif 

      #ifdef __HIP_PLATFORM_HCC___
      return __ffsll( mask);
      #endif 
   }

   /*
    * Wraps over __ffs for AMD and NVIDIA
    */
   template <typename T>
   __device__  __forceinline__
   int warpVoteAny(bool predicate,T votingMask=T(-1)){
      #ifdef __NVCC__
      return __any_sync(votingMask,predicate);
      #endif 

      #ifdef __HIP_PLATFORM_HCC___
      return __any(predicate);
      #endif 
   }

   /*
    * Wraps over __popc for AMD and NVIDIA
    */
   template <typename T>
   __device__  __forceinline__
   uint32_t pop_count(T mask){
      #ifdef __NVCC__
         return __popc(mask);
      #endif 
      #ifdef __HIP_PLATFORM_HCC___
      if constexpr(sizeof(T)==4){
         return __popc(mask);
      }else if constexpr(sizeof(mask)==8){
         return __popcll(mask);
      }else{
         //Cannot be static_assert(false...);
         static_assert(!sizeof(T*), "Not supported");
      }
      #endif 
   }

}


#endif

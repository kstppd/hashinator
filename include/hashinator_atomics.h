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

}


#endif

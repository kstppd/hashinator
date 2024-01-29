/* File:    split_allocators.h
 * Authors: Kostis Papadakis (2023)
 * Description: Custom allocators for splitvector
 *
 * This file defines the following classes:
 *    --split::split_unified_allocator;
 *    --split::split_host_allocator;
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
#include "archMacros.h"
#include "gpu_wrappers.h"
#include <cassert>
namespace split {

#ifndef SPLIT_CPU_ONLY_MODE

#ifdef __NVCC__
/* Define the CUDA error checking macro */
#define SPLIT_CHECK_ERR(err) (split::cuda_error(err, __FILE__, __LINE__))
static void cuda_error(cudaError_t err, const char* file, int line) {
   if (err != cudaSuccess) {
      printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      abort();
   }
}
#endif
#ifdef __HIP__
/* Define the HIP error checking macro */
#define SPLIT_CHECK_ERR(err) (split::hip_error(err, __FILE__, __LINE__))
static void hip_error(hipError_t err, const char* file, int line) {
   if (err != hipSuccess) {
      printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
      abort();
   }
}
#endif

/**
 * @brief Custom allocator for unified memory (GPU and CPU accessible).
 *
 * This class provides an allocator for unified memory, which can be accessed
 * by both the GPU and the CPU. It allocates and deallocates memory using split_gpuMallocManaged
 * and split_gpuFree functions, while also providing constructors and destructors for objects.
 *
 * @tparam T Type of the allocated objects.
 */
template <class T>
class split_unified_allocator {
public:
   typedef T value_type;
   typedef value_type* pointer;
   typedef const value_type* const_pointer;
   typedef value_type& reference;
   typedef const value_type& const_reference;
   typedef ptrdiff_t difference_type;
   typedef size_t size_type;
   template <class U>
   struct rebind {
      typedef split_unified_allocator<U> other;
   };
   /**
    * @brief Default constructor.
    */
   split_unified_allocator() throw() {}

   /**
    * @brief Copy constructor with different type.
    */
   template <class U>
   split_unified_allocator(split_unified_allocator<U> const&) throw() {}
   pointer address(reference x) const { return &x; }
   const_pointer address(const_reference x) const { return &x; }

   pointer allocate(size_type n, const void* /*hint*/ = 0) {
      T* ret;
      assert(n && "allocate 0");
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n * sizeof(value_type)));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   static void* allocate_raw(size_type n, const void* /*hint*/ = 0) {
      void* ret;
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   void deallocate(pointer p, size_type) { SPLIT_CHECK_ERR(split_gpuFree(p)); }

   static void deallocate(void* p, size_type) { SPLIT_CHECK_ERR(split_gpuFree(p)); }

   size_type max_size() const throw() {
      size_type max = static_cast<size_type>(-1) / sizeof(value_type);
      return (max > 0 ? max : 1);
   }

   template <typename U, typename... Args>
   __host__ __device__ void construct(U* p, Args&&... args) {
      ::new (p) U(std::forward<Args>(args)...);
   }

   void destroy(pointer p) { p->~value_type(); }
};

#endif

/**
 * @brief Custom allocator for host memory.
 *
 * This class provides an allocator for host memory, which can be accessed
 * by the CPU. It allocates and deallocates memory using malloc and free functions,
 * while also providing constructors and destructors for objects.
 *
 * @tparam T Type of the allocated objects.
 */
template <class T>
class split_host_allocator {
public:
   typedef T value_type;
   typedef value_type* pointer;
   typedef const value_type* const_pointer;
   typedef value_type& reference;
   typedef const value_type& const_reference;
   typedef ptrdiff_t difference_type;
   typedef size_t size_type;
   template <class U>
   struct rebind {
      typedef split_host_allocator<U> other;
   };

   /**
    * @brief Default constructor.
    */
   split_host_allocator() throw() {}

   /**
    * @brief Copy constructor with different type.
    */
   template <class U>
   split_host_allocator(split_host_allocator<U> const&) throw() {}
   pointer address(reference x) const { return &x; }
   const_pointer address(const_reference x) const { return &x; }

   pointer allocate(size_type n, const void* /*hint*/ = 0) {
      pointer const ret = reinterpret_cast<pointer>(malloc(n * sizeof(value_type)));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   static void* allocate_raw(size_type n, const void* /*hint*/ = 0) {
      void* ret = (void*)malloc(n);
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   void deallocate(pointer p, size_type) { free(p); }

   static void deallocate(void* p, size_type) { free(p); }

   size_type max_size() const throw() {
      size_type max = static_cast<size_type>(-1) / sizeof(value_type);
      return (max > 0 ? max : 1);
   }

   template <typename U, typename... Args>
   void construct(U* p, Args&&... args) {
      ::new (p) U(std::forward<Args>(args)...);
   }

   void destroy(pointer p) { p->~value_type(); }
};
} // namespace split

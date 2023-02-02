/* File:    split_allocators.h
 * Authors: Kostis Papadakis (2023)
 * Description: Custom allocators for splitvectors 
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

#ifdef __NVCC__
#ifdef CUDAVEC
   #define CheckErrors(msg) \
      do { \
         cudaError_t __err = cudaGetLastError(); \
         if (__err != cudaSuccess) { \
               fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                  msg, cudaGetErrorString(__err), \
                  __FILE__, __LINE__); \
               fprintf(stderr, "***** FAILED - ABORTING*****\n"); \
               exit(1); \
         } \
      } while (0)
#else
//TODO--> make it do smth.
   #define CheckErrors(msg) \
      do { }  while (0)
#endif
#endif

namespace split{
 
#ifdef __NVCC__
   template <class T>
   class split_unified_allocator{
      public:
      typedef T value_type;
      typedef value_type* pointer;
      typedef const value_type* const_pointer;
      typedef value_type& reference;
      typedef const value_type& const_reference;
      typedef ptrdiff_t difference_type;
      typedef size_t size_type;
      template<class U> struct rebind {typedef split_unified_allocator<U> other;};
      split_unified_allocator() throw() { }
      split_unified_allocator(split_unified_allocator const&) throw() { }
      template<class U>
      split_unified_allocator(split_unified_allocator<U> const&) throw() { }
      pointer address(reference x) const { return &x; }
      const_pointer address(const_reference x) const { return &x; }

      pointer allocate(size_type n, const void* /*hint*/ = 0){
         T* ret;
         cudaMallocManaged((void**)&ret, n * sizeof(value_type));
         CheckErrors("Managed Allocation");
         if (ret == nullptr) {throw std::bad_alloc();}
         return ret;
      }

      void deallocate(pointer p, size_type){
         cudaFree(p);
         CheckErrors("Managed Deallocation");
      }

      size_type max_size() const throw(){
         size_type max = static_cast<size_type>(-1) / sizeof (value_type);
         return (max > 0 ? max : 1);
      }

      template <typename U, typename... Args>
      void construct(U *p, Args&& ... args){
            ::new(p) U(std::forward<Args>(args)...);
      }

      void destroy(pointer p) { 
         p->~value_type(); 
      }
   };

#endif

   template <class T>
   class split_host_allocator{
      public:
      typedef T value_type;
      typedef value_type* pointer;
      typedef const value_type* const_pointer;
      typedef value_type& reference;
      typedef const value_type& const_reference;
      typedef ptrdiff_t difference_type;
      typedef size_t size_type;
      template<class U> struct rebind {typedef split_host_allocator<U> other;};
      split_host_allocator() throw() { }
      split_host_allocator(split_host_allocator const&) throw() { }
      template<class U>
      split_host_allocator(split_host_allocator<U> const&) throw() { }
      pointer address(reference x) const { return &x; }
      const_pointer address(const_reference x) const { return &x; }

      pointer allocate(size_type n, const void* /*hint*/ = 0){
         pointer const ret = reinterpret_cast<pointer>(malloc(n*sizeof(value_type)));
         if (ret == nullptr) {throw std::bad_alloc();}
         return ret;
      }

      void deallocate(pointer p, size_type){
         free(p);
      }

      size_type max_size() const throw(){
         size_type max = static_cast<size_type>(-1) / sizeof (value_type);
         return (max > 0 ? max : 1);
      }

      template <typename U, typename... Args>
      void construct(U *p, Args&& ... args){
            ::new(p) U(std::forward<Args>(args)...);
      }

      void destroy(pointer p) { 
         p->~value_type(); 
      }
   };
}//namespace split


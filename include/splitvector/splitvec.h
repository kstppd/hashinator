/* File:    splitvec.h
 * Authors: Kostis Papadakis (2023)
 * Description: A lightweight vector implementation that uses
 *              unified memory to easily handle data on CPUs
 *              and GPUs taking away the burden of data migration.
 *
 * This file defines the following classes:
 *    --split::SplitVector;
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
#include "split_allocators.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdlib.h>

#ifndef SPLIT_CPU_ONLY_MODE
#ifdef __NVCC__
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#define HOSTONLY __host__
#define DEVICEONLY __device__
#define HOSTDEVICE __host__ __device__
template <typename T>
using DefaultAllocator = split::split_unified_allocator<T>;
#else
#define HOSTONLY
#define DEVICEONLY
#define HOSTDEVICE
template <typename T>
using DefaultAllocator = split::split_host_allocator<T>;
#endif

namespace split {

template <typename T>
void swap(T& t1, T& t2) {
   T tmp = std::move(t1);
   t1 = std::move(t2);
   t2 = std::move(tmp);
}

/**
 * @brief Information about the SplitVector.
 *
 * This struct holds information about the size and capacity of a SplitVector instance.
 */
typedef struct SplitVectorInfo {
   size_t size;
   size_t capacity;
} SplitInfo;

enum class Residency { host, device };

/**
 * @brief A lightweight vector implementation with unified memory support.
 *
 * The SplitVector class provides a vector-like interface for managing data using unified memory,
 * allowing seamless data handling on both CPUs and GPUs without explicit migration.
 *
 * @tparam T Type of the elements in the vector.
 * @tparam Allocator The allocator type for managing memory.
 */
template <typename T, class Allocator = DefaultAllocator<T>>
class SplitVector {

private:
   T* _data = nullptr;           // actual pointer to our data
   size_t* _size;                // number of elements in vector.
   size_t* _capacity;            // number of allocated elements
   size_t _alloc_multiplier = 2; // host variable; multiplier for  when reserving more space
   Allocator _allocator;         // Allocator used to allocate and deallocate memory;
   Residency _location;          // Flags that describes the current residency of our data

   /**
    * @brief Checks if a pointer is valid and throws an exception if it's null.
    * @param ptr Pointer to be checked.
    */
   inline void _check_ptr(void* ptr) { assert(ptr); }

   /**
    * @brief Internal range check used in the .at() method.
    *
    * @param index Index to be checked.
    * @throws std::out_of_range If the index is out of range.
    */
   HOSTDEVICE void _rangeCheck(size_t index) const noexcept {
      if (index >= size()) {
         printf("Tried indexing %d/%d\n", (int)index, (int)size());
      }
      assert(index < size() && "out of range ");
   }

   /**
    * @brief Allocates memory for the vector on the host.
    *
    * @param size Number of elements to allocate.
    * @throws std::bad_alloc If memory allocation fails.
    */
   HOSTONLY void _allocate(size_t size) {
      _size = _allocate_and_construct(size);
      _capacity = _allocate_and_construct(size);
      _check_ptr(_size);
      _check_ptr(_capacity);
      if (size == 0) {
         return;
      }
      _data = _allocate_and_construct(size, T());
      _check_ptr(_data);
      if (_data == nullptr) {
         _deallocate();
         throw std::bad_alloc();
      }
   }

   /**
    * @brief Deallocates memory for the vector on the host.
    */
   HOSTONLY void _deallocate() {
      if (_data != nullptr) {
         _deallocate_and_destroy(capacity(), _data);
         _data = nullptr;
      }
      _deallocate_and_destroy(_capacity);
      _deallocate_and_destroy(_size);
   }

   /**
    * @brief Allocates memory and constructs elements on the host.
    *
    * @param n Number of elements to allocate and construct.
    * @param val Value to be used for construction.
    * @return Pointer to the allocated and constructed memory.
    */
   HOSTONLY T* _allocate_and_construct(size_t n, const T& val) {
      T* _ptr = _allocator.allocate(n);
      for (size_t i = 0; i < n; i++) {
         _allocator.construct(&_ptr[i], val);
      }
      return _ptr;
   }

   /**
    * @brief Allocates memory and constructs metadata on the host.
    *
    * @param val Value to be used for construction.
    * @return Pointer to the allocated and constructed memory.
    */
   HOSTONLY size_t* _allocate_and_construct(const size_t& val) {
      size_t* _ptr = (size_t*)_allocator.allocate_raw(sizeof(size_t));
      assert(_ptr);
      *_ptr = val;
      return _ptr;
   }

   /**
    * @brief Deallocates memory and destroys elements on the host.
    *
    * @param n Number of elements to deallocate and destroy.
    * @param _ptr Pointer to the memory to be deallocated and destroyed.
    */
   HOSTONLY void _deallocate_and_destroy(size_t n, T* _ptr) {
      for (size_t i = 0; i < n; i++) {
         _allocator.destroy(&_ptr[i]);
      }
      _allocator.deallocate(_ptr, n);
   }

   /**
    * @brief Deallocates memory for metadata on the host.
    *
    * @param ptr Pointer to the memory to be deallocated.
    */
   HOSTONLY void _deallocate_and_destroy(size_t* ptr) {
      if (ptr) {
         _allocator.deallocate(ptr, 1);
      }
   }

public:
   /* Available Constructors :
    *    -- SplitVector()                       --> Default constructor. Almost a no OP but _size and _capacity have  to
    * be allocated for device usage.
    *    -- SplitVector(size_t)                 --> Instantiates a splitvector with a specific size. (capacity == size)
    *    -- SplitVector(size_t,T)               --> Instantiates a splitvector with a specific size and sets all
    * elements to T.(capacity == size)
    *    -- SplitVector(SplitVector&)           --> Copy constructor.
    *    -- SplitVector(SplitVector&&)          --> Move constructor.
    *    -- SplitVector(std::initializer_list&) --> Creates a SplitVector and copies over the elemets of the init. list.
    *    -- SplitVector(std::vector&)           --> Creates a SplitVector and copies over the elemets of the std vector
    * */

   /**
    * @brief Default constructor. Creates an empty SplitVector.
    */
   HOSTONLY explicit SplitVector() : _location(Residency::host) {
      this->_allocate(0); // seems counter-intuitive based on stl but it is not!
   }

   /**
    * @brief Constructor to create a SplitVector of a specified size.
    *
    * @param size The size of the SplitVector to be created.
    */
   HOSTONLY explicit SplitVector(size_t size) : _location(Residency::host) { this->_allocate(size); }

   /**
    * @brief Constructor to create a SplitVector of a specified size with initial values.
    *
    * @param size The size of the SplitVector to be created.
    * @param val The initial value to be assigned to each element.
    */
   HOSTONLY explicit SplitVector(size_t size, const T& val) : _location(Residency::host) {
      this->_allocate(size);
      for (size_t i = 0; i < size; i++) {
         _data[i] = val;
      }
   }

   /**
    * @brief Copy constructor to create a SplitVector from another SplitVector.
    *
    * @param other The SplitVector to be copied.
    */
#ifdef SPLIT_CPU_ONLY_MODE
   HOSTONLY explicit SplitVector(const SplitVector<T, Allocator>& other) {
      const size_t size_to_allocate = other.size();
      this->_allocate(size_to_allocate);
      for (size_t i = 0; i < size_to_allocate; i++) {
         _data[i] = other._data[i];
      }
   }
#else

   HOSTONLY explicit SplitVector(const SplitVector<T, Allocator>& other) {
      const size_t size_to_allocate = other.size();
      auto copySafe = [&]() -> void {
         for (size_t i = 0; i < size_to_allocate; i++) {
            _data[i] = other._data[i];
         }
      };
      this->_allocate(size_to_allocate);
      if constexpr (std::is_trivially_copyable<T>::value) {
         if (other._location == Residency::device) {
            optimizeGPU();
            SPLIT_CHECK_ERR(
                split_gpuMemcpy(_data, other._data, size_to_allocate * sizeof(T), split_gpuMemcpyDeviceToDevice));
            return;
         }
      }
      copySafe();
      _location = Residency::host;
   }
#endif
   /**
    * @brief Move constructor to move from another SplitVector.
    *
    * @param other The SplitVector to be moved from.
    */
   HOSTONLY SplitVector(SplitVector<T, Allocator>&& other) noexcept {
      _data = other._data;
      *_size = other.size();
      *_capacity = other.capacity();
      *(other._capacity) = 0;
      *(other._size) = 0;
      other._data = nullptr;
      _location = other._location;
   }

   /**
    * @brief Constructor to create a SplitVector from an initializer list.
    *
    * @param init_list The initializer list to initialize the SplitVector with.
    */
   HOSTONLY explicit SplitVector(std::initializer_list<T> init_list) : _location(Residency::host) {
      this->_allocate(init_list.size());
      for (size_t i = 0; i < size(); i++) {
         _data[i] = init_list.begin()[i];
      }
   }

   /**
    * @brief Constructor to create a SplitVector from a std::vector.
    *
    * @param other The std::vector to initialize the SplitVector with.
    */
   HOSTONLY explicit SplitVector(const std::vector<T>& other) : _location(Residency::host) {
      this->_allocate(other.size());
      for (size_t i = 0; i < size(); i++) {
         _data[i] = other[i];
      }
   }

   /**
    * @brief Destructor for the SplitVector. Deallocates memory.
    */
   HOSTONLY ~SplitVector() { _deallocate(); }

/**
 * @brief Custom assignment operator to assign the content of another SplitVector.
 *
 * @param other The SplitVector to assign from.
 * @return Reference to the assigned SplitVector.
 */
#ifdef SPLIT_CPU_ONLY_MODE
   HOSTONLY SplitVector<T, Allocator>& operator=(const SplitVector<T, Allocator>& other) {
      // Match other's size prior to copying
      resize(other.size());
      for (size_t i = 0; i < other.size(); i++) {
         _data[i] = other._data[i];
      }
      return *this;
   }
#else

   HOSTONLY SplitVector<T, Allocator>& operator=(const SplitVector<T, Allocator>& other) {
      // Match other's size prior to copying
      resize(other.size());
      auto copySafe = [&]() -> void {
         for (size_t i = 0; i < size(); i++) {
            _data[i] = other._data[i];
         }
      };

      if constexpr (std::is_trivially_copyable<T>::value) {
         if (other._location == Residency::device) {
            optimizeGPU();
            SPLIT_CHECK_ERR(split_gpuMemcpy(_data, other._data, size() * sizeof(T), split_gpuMemcpyDeviceToDevice));
            return *this;
         }
      }
      copySafe();
      _location = Residency::host;
      return *this;
   }
#endif

   /**
    * @brief Move assignment operator to move from another SplitVector.
    *
    * @param other The SplitVector to move from.
    * @return Reference to the moved SplitVector.
    */
   HOSTONLY SplitVector<T, Allocator>& operator=(SplitVector<T, Allocator>&& other) noexcept {
      if (this == &other) {
         return *this;
      }

      _deallocate_and_destroy(capacity(), _data);
      _data = other._data;
      *_size = other.size();
      *_capacity = other.capacity();
      *(other._capacity) = 0;
      *(other._size) = 0;
      other._data = nullptr;
      _location = other._location;
      return *this;
   }

   /**
    * @brief Custom new operator for allocation using the allocator.
    *
    * @param len The size to allocate.
    * @return Pointer to the allocated memory.
    */
   HOSTONLY
   void* operator new(size_t len) {
      void* ptr = Allocator::allocate_raw(len);
      return ptr;
   }

   /**
    * @brief Custom delete operator for deallocation using the allocator.
    *
    * @param ptr Pointer to the memory to deallocate.
    */
   HOSTONLY
   void operator delete(void* ptr) { Allocator::deallocate(ptr, 1); }

   /**
    * @brief Custom new operator for array allocation using the allocator.
    *
    * @param len The size to allocate.
    * @return Pointer to the allocated memory.
    */
   HOSTONLY
   void* operator new[](size_t len) {
      void* ptr = Allocator::allocate_raw(len);
      return ptr;
   }

   /**
    * @brief Custom delete operator for array deallocation using the allocator.
    *
    * @param ptr Pointer to the memory to deallocate.
    */
   HOSTONLY
   void operator delete[](void* ptr) { Allocator::deallocate(ptr); }

#ifndef SPLIT_CPU_ONLY_MODE
   /**
    * @brief Uploads the SplitVector to the GPU.
    *
    * @param stream The GPU stream to perform the upload on.
    * @return Pointer to the uploaded SplitVector on the GPU.
    * Has to be split_gpuFree'd after use otherwise memleak (small one but still)!
    */
   HOSTONLY
   SplitVector<T, Allocator>* upload(split_gpuStream_t stream = 0) {
      SplitVector* d_vec;
      optimizeGPU(stream);
      SPLIT_CHECK_ERR(split_gpuMallocAsync((void**)&d_vec, sizeof(SplitVector), stream));
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(d_vec, this, sizeof(SplitVector), split_gpuMemcpyHostToDevice, stream));
      return d_vec;
   }

   /**
    * @brief Manually prefetches data to the GPU.
    *
    * @param stream The GPU stream to perform the prefetch on.
    */
   HOSTONLY void optimizeGPU(split_gpuStream_t stream = 0) noexcept {
      if (_location == Residency::device) {
         return;
      }
      _location = Residency::device;
      int device;
      SPLIT_CHECK_ERR(split_gpuGetDevice(&device));

      // First make sure _capacity does not page-fault ie prefetch it to host
      // This is done because _capacity would page-fault otherwise as pointed by Markus
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_capacity, sizeof(size_t), split_gpuCpuDeviceId, stream));
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(stream));

      // Now prefetch everything to device
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_data, capacity() * sizeof(T), device, stream));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_size, sizeof(size_t), device, stream));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_capacity, sizeof(size_t), device, stream));
   }

   /**
    * @brief Manually prefetches data to the CPU.
    *
    * @param stream The GPU stream to perform the prefetch on.
    */
   HOSTONLY void optimizeCPU(split_gpuStream_t stream = 0) noexcept {
      if (_location == Residency::host) {
         return;
      }
      _location = Residency::host;
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_capacity, sizeof(size_t), split_gpuCpuDeviceId, stream));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_size, sizeof(size_t), split_gpuCpuDeviceId, stream));
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(stream));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_data, capacity() * sizeof(T), split_gpuCpuDeviceId, stream));
   }

   /**
    * @brief Attaches the SplitVector to a specific GPU stream.
    *
    * @param s The GPU stream to attach to.
    * @param flags Flags for memory attachment.
    */
   HOSTONLY void streamAttach(split_gpuStream_t s, uint32_t flags = split_gpuMemAttachSingle) {
      SPLIT_CHECK_ERR(split_gpuStreamAttachMemAsync(s, (void*)_size, sizeof(size_t), flags));
      SPLIT_CHECK_ERR(split_gpuStreamAttachMemAsync(s, (void*)_capacity, sizeof(size_t), flags));
      SPLIT_CHECK_ERR(split_gpuStreamAttachMemAsync(s, (void*)_data, *_capacity * sizeof(T), flags));
      return;
   }

   /**
    * @brief Copies metadata to a provided destination SplitInfo structure.
    *
    * @param dst Pointer to the destination SplitInfo structure.
    * @param s The GPU stream to perform the copy on.
    */
   HOSTONLY void copyMetadata(SplitInfo* dst, split_gpuStream_t s = 0) {
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&dst->size, _size, sizeof(size_t), split_gpuMemcpyDeviceToHost, s));
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&dst->capacity, _capacity, sizeof(size_t), split_gpuMemcpyDeviceToHost, s));
   }

   /**
    * @brief Passes memory advice directives to the data.
    *
    * @param advice The memory advice to be passed.
    * @param device The GPU device to target.
    * @param stream The GPU stream to perform the operation on.
    */
   HOSTONLY void memAdvise(split_gpuMemoryAdvise advice, int device = -1, split_gpuStream_t stream = 0) {
      if (device == -1) {
         SPLIT_CHECK_ERR(split_gpuGetDevice(&device));
      }
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_capacity, sizeof(size_t), split_gpuCpuDeviceId, stream));
      SPLIT_CHECK_ERR(split_gpuStreamSynchronize(stream));
      SPLIT_CHECK_ERR(split_gpuMemAdvise(_data, capacity() * sizeof(T), advice, device));
      SPLIT_CHECK_ERR(split_gpuMemAdvise(_size, sizeof(size_t), advice, device));
      SPLIT_CHECK_ERR(split_gpuMemAdvise(_capacity, sizeof(size_t), advice, device));
      SPLIT_CHECK_ERR(split_gpuMemPrefetchAsync(_capacity, sizeof(size_t), device, stream));
   }
#endif

   /**
    * @brief Swaps the content of two SplitVectors.
    *
    * @param other The other SplitVector to swap with.
    * Pointers outside of splitvector's source
    * are invalidated after swap is called.
    */
   void swap(SplitVector<T, Allocator>& other) noexcept {
      if (*this == other) { // no need to do any work
         return;
      }
      split::swap(_data, other._data);
      split::swap(_size, other._size);
      split::swap(_capacity, other._capacity);
      split::swap(_allocator, other._allocator);
      return;
   }

   /************STL compatibility***************/
   /**
    * @brief Returns the number of elements in the container.
    *
    * @return Number of elements in the container.
    */
   HOSTDEVICE const size_t& size() const noexcept { return *_size; }

   /**
    * @brief Bracket accessor for accessing elements by index without bounds check.
    *
    * @param index The index of the element to access.
    * @return Reference to the accessed element.
    */
   HOSTDEVICE T& operator[](size_t index) noexcept { return _data[index]; }

   /**
    * @brief Const bracket accessor for accessing elements by index without bounds check.
    *
    * @param index The index of the element to access.
    * @return Const reference to the accessed element.
    */
   HOSTDEVICE const T& operator[](size_t index) const noexcept { return _data[index]; }

   /**
    * @brief At accessor with bounds check for accessing elements by index.
    *
    * @param index The index of the element to access.
    * @return Reference to the accessed element.
    */
   HOSTDEVICE T& at(size_t index) {
      _rangeCheck(index);
      return _data[index];
   }

   /**
    * @brief Const at accessor with bounds check for accessing elements by index.
    *
    * @param index The index of the element to access.
    * @return Const reference to the accessed element.
    */
   HOSTDEVICE const T& at(size_t index) const {
      _rangeCheck(index);
      return _data[index];
   }

   /**
    * @brief Returns a raw pointer to the data stored in the SplitVector.
    *
    * @return Pointer to the data.
    */
   HOSTDEVICE T* data() noexcept { return &(_data[0]); }

   /**
    * @brief Returns a const raw pointer to the data stored in the SplitVector.
    *
    * @return Const pointer to the data.
    */
   HOSTDEVICE const T* data() const noexcept { return &(_data[0]); }

   /**
    * @brief Reallocates data to a bigger chunk of memory.
    *
    * @param requested_space The size of the requested space.
    */
   HOSTONLY void reallocate(size_t requested_space) {
      if (requested_space == 0) {
         if (_data != nullptr) {
            _deallocate_and_destroy(capacity(), _data);
         }
         _data = nullptr;
         *_capacity = 0;
         *_size = 0;
         return;
      }
      T* _new_data;
      _new_data = _allocate_and_construct(requested_space, T());
      if (_new_data == nullptr) {
         _deallocate_and_destroy(requested_space, _new_data);
         this->_deallocate();
         throw std::bad_alloc();
      }

      // Copy over
      for (size_t i = 0; i < size(); i++) {
         _new_data[i] = _data[i];
      }

      // Deallocate old space
      _deallocate_and_destroy(capacity(), _data);

      // Swap pointers & update capacity
      // Size remains the same ofc
      _data = _new_data;
      *_capacity = requested_space;
      return;
   }

   /**
    * @brief Reserves memory for the SplitVector.
    *
    * @param requested_space The size of the requested space.
    * @param eco Indicates whether to allocate exactly the requested space.
    * Supports only host reserving.
    * Will never reduce the vector's size.
    * Memory location will change so any old pointers/iterators
    * will be invalidated after a call.
    */
   HOSTONLY
   void reserve(size_t requested_space, bool eco = false) {
      size_t current_space = *_capacity;
      // Vector was default initialized
      if (_data == nullptr) {
         _deallocate();
         _allocate(requested_space);
         *_size = 0;
         return;
      }
      // Nope.
      if (requested_space <= current_space) {
         for (size_t i = size(); i < requested_space; ++i) {
            _allocator.construct(&_data[i], T());
         }
         return;
      }
      // If the users passes eco=true we allocate
      // exactly what was requested
      if (!eco) {
         requested_space *= _alloc_multiplier;
      }
      reallocate(requested_space);
      return;
   }

   /**
    * @brief Resize the SplitVector to a new size.
    *
    * @param newSize The new size of the SplitVector.
    * @param eco Indicates whether to allocate exactly the requested space.
    * Supports only host resizing.
    * If new size is smaller than the current size we just reduce size but
    * the capacity remains the same
    * Memory location will change so any old pointers/iterators
    * will be invalid from now on.
    */
   HOSTONLY
   void resize(size_t newSize, bool eco = false) {
      // Let's reserve some space and change our size
      if (newSize <= size()) {
         *_size = newSize;
         return;
      }
      reserve(newSize, eco);
      *_size = newSize;
   }

#ifndef SPLIT_CPU_ONLY_MODE
   /**
    * @brief Resize the SplitVector on the device.
    *
    * @param newSize The new size of the SplitVector.
    */
   DEVICEONLY
   void device_resize(size_t newSize) {
      if (newSize > capacity()) {
         assert(0 && "Splitvector has a catastrophic failure trying to resize on device.");
      }
      for (size_t i = size(); i < newSize; ++i) {
         _allocator.construct(&_data[i], T());
      }
      *_size = newSize;
   }
#endif

   /**
    * @brief Increase the capacity of the SplitVector by 1.
    */
   HOSTONLY
   void grow() { reserve(capacity() + 1); }

   /**
    * @brief Reduce the capacity of the SplitVector to match its size.
    */
   HOSTONLY
   void shrink_to_fit() {
      size_t curr_cap = *_capacity;
      size_t curr_size = *_size;

      if (curr_cap == curr_size) {
         return;
      }

      reallocate(curr_size);
      return;
   }

   /**
    * @brief Remove the last element from the SplitVector.
    */
   HOSTDEVICE
   void pop_back() {
      if (size() > 0) {
         remove_from_back(1);
      }
      return;
   }

   /**
    * @brief Remove n elements from the back of the SplitVector.
    *
    * @param n The number of elements to remove.
    */
   HOSTDEVICE
   void remove_from_back(size_t n) noexcept {
      const size_t end = size() - n;
      if constexpr (std::is_nothrow_destructible<T>::value) {
         for (auto i = size(); i > end;) {
            (_data + --i)->~T();
         }
      }
      *_size = end;
   }

   /**
    * @brief Clear all elements from the SplitVector.
    */
   HOSTDEVICE
   void clear() noexcept {
      if constexpr (std::is_nothrow_destructible<T>::value) {
         for (size_t i = 0; i < size(); i++) {
            _data[i].~T();
         }
      }
      *_size = 0;
      return;
   }

   /**
    * @brief Get the current capacity of the SplitVector.
    *
    * @return The capacity of the SplitVector.
    */
   HOSTDEVICE
   inline size_t capacity() const noexcept { return *_capacity; }

   /**
    * @brief Get a reference to the last element of the SplitVector.
    *
    * @return Reference to the last element.
    */
   HOSTDEVICE
   T& back() noexcept { return _data[*_size - 1]; }

   /**
    * @brief Get a const reference to the last element of the SplitVector.
    *
    * @return Const reference to the last element.
    */
   HOSTDEVICE
   const T& back() const noexcept { return _data[*_size - 1]; }

   /**
    * @brief Get a reference to the first element of the SplitVector.
    *
    * @return Reference to the first element.
    */
   HOSTDEVICE
   T& front() noexcept { return _data[0]; }

   /**
    * @brief Get a const reference to the first element of the SplitVector.
    *
    * @return Const reference to the first element.
    */
   HOSTDEVICE
   const T& front() const noexcept { return _data[0]; }

   /**
    * @brief Check if the SplitVector is empty.
    *
    * @return True if the SplitVector is empty, otherwise false.
    */
   HOSTDEVICE
   bool empty() const noexcept { return size() == 0; }

   /**
    * @brief Push an element to the back of the SplitVector.
    *
    * @param val The value to push to the back.
    */
   HOSTONLY
   void push_back(const T& val) {
      // If we have no allocated memory because the default ctor was used then
      // allocate one element, set it and return
      if (_data == nullptr) {
         *this = SplitVector<T, Allocator>(1, val);
         return;
      }
      resize(size() + 1);
      _data[size() - 1] = val;
      return;
   }

   /**
    * @brief Push a moved element to the back of the SplitVector.
    *
    * @param val The value to push to the back.
    */
   HOSTONLY
   void push_back(const T&& val) {

      // If we have no allocated memory because the default ctor was used then
      // allocate one element, set it and return
      if (_data == nullptr) {
         *this = SplitVector<T, Allocator>(1, std::move(val));
         return;
      }
      resize(size() + 1);
      _data[size() - 1] = std::move(val);
      return;
   }

#ifndef SPLIT_CPU_ONLY_MODE
   /**
    * @brief Push an element to the back of the SplitVector on the device.
    *
    * @param val The value to push to the back.
    */
   DEVICEONLY
   bool device_push_back(const T& val) {
      size_t old = atomicAdd((unsigned int*)_size, 1);
      if (old >= capacity() - 1) {
         atomicSub((unsigned int*)_size, 1);
         return false;
      }
      atomicCAS(&(_data[old]), _data[old], val);
      return true;
   }

   /**
    * @brief Push a moved element to the back of the SplitVector on the device.
    *
    * @param val The value to push to the back.
    */
   DEVICEONLY
   bool device_push_back(const T&& val) {

      // We need at least capacity=size+1 otherwise this
      // pushback cannot be done
      size_t old = atomicAdd((unsigned int*)_size, 1);
      if (old >= capacity() - 1) {
         atomicSub((unsigned int*)_size, 1);
         return false;
      }
      atomicCAS(&(_data[old]), _data[old], std::move(val));
      return true;
   }
#endif

   // Iterators
   class iterator {

   private:
      T* _data;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = int64_t;
      using pointer = T*;
      using reference = T&;

      // iterator(){}
      HOSTDEVICE
      iterator(pointer data) : _data(data) {}

      HOSTDEVICE
      pointer data() { return _data; }
      HOSTDEVICE
      pointer operator->() { return _data; }
      HOSTDEVICE
      reference operator*() { return *_data; }

      HOSTDEVICE
      bool operator==(const iterator& other) const { return _data == other._data; }
      HOSTDEVICE
      bool operator!=(const iterator& other) const { return _data != other._data; }
      HOSTDEVICE
      iterator& operator++() {
         _data += 1;
         return *this;
      }
      HOSTDEVICE
      iterator operator++(int) { return iterator(_data + 1); }
      HOSTDEVICE
      iterator operator--(int) { return iterator(_data - 1); }
      HOSTDEVICE
      iterator operator--() {
         _data -= 1;
         return *this;
      }
      HOSTDEVICE
      iterator& operator+=(int64_t offset) {
         _data += offset;
         return *this;
      }
      HOSTDEVICE
      iterator& operator-=(int64_t offset) {
         _data -= offset;
         return *this;
      }
      HOSTDEVICE
      iterator operator+(int64_t offset) const {
         iterator itt(*this);
         return itt += offset;
      }
      HOSTDEVICE
      iterator operator-(int64_t offset) const {
         iterator itt(*this);
         return itt -= offset;
      }
   };

   class const_iterator {

   private:
      const T* _data;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = int64_t;
      using pointer = const T*;
      using reference = const T&;

      HOSTDEVICE
      const_iterator(pointer data) : _data(data) {}

      HOSTDEVICE
      pointer data() const { return _data; }
      HOSTDEVICE
      pointer operator->() const { return _data; }
      HOSTDEVICE
      reference operator*() const { return *_data; }

      HOSTDEVICE
      bool operator==(const const_iterator& other) const { return _data == other._data; }
      HOSTDEVICE
      bool operator!=(const const_iterator& other) const { return _data != other._data; }
      HOSTDEVICE
      const_iterator& operator++() {
         _data += 1;
         return *this;
      }
      HOSTDEVICE
      const_iterator operator++(int) { return const_iterator(_data + 1); }
      HOSTDEVICE
      const_iterator operator--(int) { return const_iterator(_data - 1); }
      HOSTDEVICE
      const_iterator operator--() {
         _data -= 1;
         return *this;
      }
      HOSTDEVICE
      const_iterator& operator+=(int64_t offset) {
         _data += offset;
         return *this;
      }
      HOSTDEVICE
      const_iterator& operator-=(int64_t offset) {
         _data -= offset;
         return *this;
      }
      HOSTDEVICE
      const_iterator operator+(int64_t offset) const {
         const_iterator itt(*this);
         return itt += offset;
      }
      HOSTDEVICE
      const_iterator operator-(int64_t offset) const {
         const_iterator itt(*this);
         return itt -= offset;
      }
   };

   /**
    * @brief Get an iterator pointing to the beginning of the vector.
    *
    * @return Iterator to the beginning of the vector.
    */
   HOSTDEVICE
   iterator begin() noexcept { return iterator(_data); }

   /**
    * @brief Get an iterator pointing to the end of the vector.
    *
    * @return Iterator to the end of the vector.
    */
   HOSTDEVICE
   const_iterator begin() const noexcept { return const_iterator(_data); }

   /**
    * @brief Get an iterator pointing to the end of the vector.
    *
    * @return Iterator to the end of the vector.
    */

   HOSTDEVICE
   iterator end() noexcept { return iterator(_data + size()); }

   /**
    * @brief Get a constant iterator pointing to the end of the vector.
    *
    * @return Constant iterator to the end of the vector.
    */
   HOSTDEVICE
   const_iterator end() const noexcept { return const_iterator(_data + size()); }

   /**
    * @brief Insert a single element at the specified position.
    *
    * @param it Iterator pointing to the position where the element should be inserted.
    * @param val The value to insert.
    * @return Iterator pointing to the inserted element.
    */
   HOSTONLY
   iterator insert(iterator it, const T& val) {

      // If empty or inserting at the end no relocating is needed
      if (it == end()) {
         push_back(val);
         return end()--;
      }

      int64_t index = it.data() - begin().data();
      if (index < 0 || index > size()) {
         throw std::out_of_range("Insert");
      }

      // Do we do need to increase our capacity?
      if (size() == capacity()) {
         grow();
      }

      for (int64_t i = size() - 1; i >= index; i--) {
         _data[i + 1] = _data[i];
      }
      _data[index] = val;
      *_size = *_size + 1;
      return iterator(_data + index);
   }

   /**
    * @brief Insert a specified number of elements with the same value at the specified position.
    *
    * @param it Iterator pointing to the position where the elements should be inserted.
    * @param elements The number of elements to insert.
    * @param val The value to insert.
    * @return Iterator pointing to the first inserted element.
    */
   HOSTONLY
   iterator insert(iterator it, const size_t elements, const T& val) {

      int64_t index = it.data() - begin().data();
      size_t oldsize = size();
      size_t newSize = oldsize + elements;
      if (index < 0 || index > size()) {
         throw std::out_of_range("Insert");
      }

      // Do we do need to increase our capacity?
      if (newSize > size()) {
         resize(newSize);
      }

      it = begin().data() + index;
      iterator last = it.data() + oldsize;
      std::copy_backward(it, last, last.data() + elements);
      last = it.data() + elements;
      std::fill(it, last, val);
      iterator retval = &_data[index];
      return retval;
   }

   /**
    * @brief Insert a range of elements at the specified position.
    *
    * @tparam InputIterator Type of the input iterator.
    * @param it Iterator pointing to the position where the elements should be inserted.
    * @param p0 Start of the input range.
    * @param p1 End of the input range.
    * @return Iterator pointing to the first inserted element.
    */
   template <typename InputIterator, class = typename std::enable_if<!std::is_integral<InputIterator>::value>::type>
   HOSTONLY iterator insert(iterator it, InputIterator p0, InputIterator p1) {

      const int64_t count = std::distance(p0, p1);
      const int64_t index = it.data() - begin().data();

      if (index < 0 || index > size()) {
         throw std::out_of_range("Insert");
      }

      size_t old_size = size();
      if (size() + count > capacity()) {
         resize(size() + count);
      }

      iterator retval = &_data[index];
      std::move(retval, iterator(&_data[old_size]), retval.data() + count);
      std::copy(p0, p1, retval);
      return retval;
   }

#ifndef SPLIT_CPU_ONLY_MODE
   /**
    * @brief Device version of insert for inserting a range of elements at the specified position.
    *
    * @tparam InputIterator Type of the input iterator.
    * @param it Iterator pointing to the position where the elements should be inserted.
    * @param p0 Start of the input range.
    * @param p1 End of the input range.
    * @return Iterator pointing to the first inserted element.
    */
   template <typename InputIterator, class = typename std::enable_if<!std::is_integral<InputIterator>::value>::type>
   DEVICEONLY iterator device_insert(iterator it, InputIterator p0, InputIterator p1) noexcept {

      const int64_t count = p1.data() - p0.data();
      const int64_t index = it.data() - begin().data();

      if (index < 0 || index > size()) {
         assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no "
                     "space available.");
      }

      if (size() + count > capacity()) {
         assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no "
                     "space available.");
      }

      // Increase size;
      device_resize(size() + count);
      for (size_t i = 0; i < count; ++i) {
         _data[index + i] = *(p0.data() + i);
      }
      iterator retval = &_data[index + count];
      return retval;
   }

   /**
    * @brief Device version of insert for inserting a single element at the specified position.
    *
    * @param it Iterator pointing to the position where the element should be inserted.
    * @param val The value to insert.
    * @return Iterator pointing to the inserted element.
    */
   DEVICEONLY
   iterator device_insert(iterator it, const T& val) noexcept {

      // If empty or inserting at the end no relocating is needed
      if (it == end()) {
         device_push_back(val);
         return end()--;
      }

      int64_t index = it.data() - begin().data();
      if (index < 0 || index > size()) {
         assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no "
                     "space available.");
      }

      if (size() == capacity()) {
         assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no "
                     "space available.");
      }

      // Increase size;
      for (int64_t i = size() - 1; i >= index; i--) {
         _data[i + 1] = _data[i];
      }
      _data[index] = val;
      device_resize(size() + 1);
      return iterator(_data + index);
   }

   /**
    * @brief Device version of insert for inserting a specified number of elements with the same value at the specified
    * position.
    *
    * @param it Iterator pointing to the position where the elements should be inserted.
    * @param elements The number of elements to insert.
    * @param val The value to insert.
    * @return Iterator pointing to the first inserted element.
    */
   DEVICEONLY
   iterator device_insert(iterator it, const size_t elements, const T& val) {

      int64_t index = it.data() - begin().data();
      size_t oldsize = size();
      size_t newSize = oldsize + elements;
      if (index < 0 || index > size()) {
         assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no "
                     "space available.");
      }

      // Do we do need to increase our capacity?
      if (newSize > size()) {
         device_resize(newSize);
      }

      it = begin().data() + index;
      iterator last = it.data() + oldsize;
      iterator target = last.data() + elements;
      for (iterator candidate = last; candidate != it; --candidate) {
         *target = *candidate;
         --target;
      }

      last = it.data() + elements;
      // std::fill(it,last,val);
      target = last;
      for (iterator candidate = it; candidate != last; ++candidate) {
         *target = val;
         ++target;
      }

      target = &_data[index];
      return target;
   }

#endif

   /**
    * @brief Erase an element at the specified position.
    *
    * @param it Iterator pointing to the element to erase.
    * @return Iterator pointing to the element after the erased one.
    */
   HOSTDEVICE
   iterator erase(iterator it) noexcept {
      const int64_t index = it.data() - begin().data();
      if constexpr (std::is_nothrow_destructible<T>::value) {
         _data[index].~T();
         for (auto i = index; i < size() - 1; i++) {
            new (&_data[i]) T(_data[i + 1]);
            _data[i + 1].~T();
         }
      } else {
         for (auto i = index; i < size() - 1; i++) {
            new (&_data[i]) T(_data[i + 1]);
         }
      }
      *_size -= 1;
      iterator retval = &_data[index];
      return retval;
   }

   /**
    * @brief Erase elements in the specified range.
    *
    * @param p0 Iterator pointing to the start of the range to erase.
    * @param p1 Iterator pointing to the end of the range to erase.
    * @return Iterator pointing to the element after the last erased one.
    */
   HOSTDEVICE
   iterator erase(iterator p0, iterator p1) noexcept {
      const int64_t start = p0.data() - begin().data();
      const int64_t end = p1.data() - begin().data();
      const int64_t offset = end - start;

      if constexpr (std::is_nothrow_destructible<T>::value) {
         for (int64_t i = 0; i < offset; i++) {
            _data[i].~T();
         }
         for (auto i = start; i < size() - offset; ++i) {
            new (&_data[i]) T(_data[i + offset]);
            _data[i + offset].~T();
         }
      } else {
         for (auto i = start; i < size() - offset; ++i) {
            new (&_data[i]) T(_data[i + offset]);
         }
      }
      *_size -= end - start;
      iterator it = &_data[start];
      return it;
   }

   HOSTONLY
   Allocator get_allocator() const noexcept { return _allocator; }

   /**
    * @brief Emplace an element at the specified position.
    *
    * @tparam Args Variadic template for constructor arguments.
    * @param pos Iterator pointing to the position where the element should be emplaced.
    * @param args Constructor arguments for the element.
    * @return Iterator pointing to the emplaced element.
    */
   template <class... Args>
   iterator emplace(iterator pos, Args&&... args) {
      const int64_t index = pos.data() - begin().data();
      if (index < 0 || index > size()) {
         throw new std::out_of_range("Out of range");
      }
      resize(size() + 1);
      iterator it = &_data[index];
      std::move(it.data(), end().data(), it.data() + 1);
      _allocator.destroy(it.data());
      _allocator.construct(it.data(), args...);
      return it;
   }

   /**
    * @brief Emplace an element at the end of the vector.
    *
    * @tparam Args Variadic template for constructor arguments.
    * @param args Constructor arguments for the element.
    */
   template <class... Args>
   void emplace_back(Args&&... args) {
      emplace(end(), std::forward<Args>(args)...);
   }
}; // SplitVector

/*Equal operator*/
template <typename T, class Allocator>
static inline HOSTDEVICE bool operator==(const SplitVector<T, Allocator>& lhs,
                                         const SplitVector<T, Allocator>& rhs) noexcept {
   if (lhs.size() != rhs.size()) {
      return false;
   }
   for (size_t i = 0; i < lhs.size(); i++) {
      if (!(lhs[i] == rhs[i])) {
         return false;
      }
   }
   // if we end up here the vectors are equal
   return true;
}

/*Not-Equal operator*/
template <typename T, class Allocator>
static inline HOSTDEVICE bool operator!=(const SplitVector<T, Allocator>& lhs,
                                         const SplitVector<T, Allocator>& rhs) noexcept {
   return !(rhs == lhs);
}

#ifndef SPLIT_CPU_ONLY_MODE
template <typename T>
class SplitDeviceVector {
   static_assert(std::is_trivially_copyable<T>::value &&
         "SplitDeviceVector only works for POD types");

private:
   enum MEMBER{SIZE,CAPACITY};

   // packed to 1 cache line
   struct __attribute__((__packed__)) Meta {
      size_t size;
      size_t capacity;
      char padding[64 - 2 * sizeof(size_t)]; // pad up to cache line
      HOSTDEVICE
      inline size_t& operator[](MEMBER member) noexcept { 
         switch (member)
         {
            case MEMBER::SIZE:
               return *reinterpret_cast<size_t*>(this); 
            case MEMBER::CAPACITY:
               return *(reinterpret_cast<size_t*>(this)+1); 
            default:
               abort();
         }
      }
   };

   // Members
   Meta* _meta = nullptr;
   T* _data = nullptr;
   split_gpuStream_t _stream;

   void setupSpace(void* ptr) noexcept {
      _meta = reinterpret_cast<Meta*>(ptr);
      _data = reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + sizeof(Meta));
   }

   [[nodiscard]] void* _allocate(const size_t sz) {
      void* _ptr = nullptr;
      if (_stream==NULL){
         SPLIT_CHECK_ERR(split_gpuMalloc((void**)&_ptr, sizeof(Meta) + sz * sizeof(T)));
      }else{
         SPLIT_CHECK_ERR(split_gpuMallocAsync((void**)&_ptr, sizeof(Meta) + sz * sizeof(T),_stream));
      }
      return _ptr;
   }

   void _deallocate(void* _ptr) {
      if (_ptr == nullptr) {
         return;
      }
      if (_stream==NULL){
         SPLIT_CHECK_ERR(split_gpuFree((void*)_ptr));
      }else{
         SPLIT_CHECK_ERR(split_gpuFreeAsync((void*)_ptr,_stream));
      }
      _ptr = nullptr;
   }

   DEVICEONLY
   inline void _rangeCheckDevice(size_t index) const noexcept {
      if (index >= _meta->size) {
         assert(true && " out of range ");
      }
   }

   HOSTONLY
   inline void _rangeCheckHost(size_t index) const noexcept {
      Meta currentMeta = getMeta();
      if (index >= currentMeta.size) {
         assert(true && " out of range ");
      }
   }

   HOSTONLY
   inline Meta getMeta() const noexcept {
      Meta buffer;
      SPLIT_CHECK_ERR(split_gpuMemcpy(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost));
      return buffer;
   }

   HOSTONLY
   inline void getMeta(Meta& buffer) const noexcept {
      SPLIT_CHECK_ERR(split_gpuMemcpy(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost));
   }

   HOSTONLY
   inline void setMeta(const Meta& buffer) noexcept {
      SPLIT_CHECK_ERR(split_gpuMemcpy(_meta, &buffer, sizeof(Meta), split_gpuMemcpyHostToDevice));
   }

   HOSTONLY
   inline T getElementFromDevice(const size_t index) const noexcept {
      T retval;
      SPLIT_CHECK_ERR(split_gpuMemcpy(&retval, &_data[index], sizeof(T), split_gpuMemcpyDeviceToHost));
      return retval;
   }

   HOSTONLY
   inline void setElementFromHost(const size_t index, const T& val) noexcept {
      SPLIT_CHECK_ERR(split_gpuMemcpy(&_data[index], &val, sizeof(T), split_gpuMemcpyHostToDevice));
      return;
   }

public:
   SplitDeviceVector(size_t sz = 0):_stream(NULL) {
      void* ptr = _allocate(sz);
      setupSpace(ptr);
      size_t s = sz;
      SPLIT_CHECK_ERR(split_gpuMemcpy(&_meta->size, &s, sizeof(size_t), split_gpuMemcpyHostToDevice));
      SPLIT_CHECK_ERR(split_gpuMemcpy(&_meta->capacity, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice));
   }

   SplitDeviceVector(size_t sz , split_gpuStream_t s):_stream(s) {
      void* ptr = _allocate(sz);
      setupSpace(ptr);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&_meta->size, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice,s));
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&_meta->capacity, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice,s));
   }

   SplitDeviceVector(const SplitDeviceVector<T>& other):_stream(NULL){
      Meta otherMeta = other.getMeta();
      void* ptr = _allocate(otherMeta.size);
      setupSpace(ptr);
      Meta newMeta{.size = otherMeta.size, .capacity = otherMeta.size};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, other._data, otherMeta.size * sizeof(T), split_gpuMemcpyDeviceToDevice));
      return;
   }

   SplitDeviceVector(const SplitDeviceVector<T>& other, split_gpuStream_t s):_stream(s){
      Meta otherMeta = other.getMeta();
      void* ptr = _allocate(otherMeta.size);
      setupSpace(ptr);
      Meta newMeta{.size = otherMeta.size, .capacity = otherMeta.size};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_data, other._data, otherMeta.size * sizeof(T), split_gpuMemcpyDeviceToDevice,s));
      return;
   }


   SplitDeviceVector(const SplitDeviceVector<T>&& other):_stream(NULL) {
      _meta = other._meta;
      _data = other._data;
      other._meta = nullptr;
      other._data = nullptr;
   }

   SplitDeviceVector(const SplitDeviceVector<T>&& other,split_gpuStream_t s):_stream(s) {
      _meta = other._meta;
      _data = other._data;
      other._meta = nullptr;
      other._data = nullptr;
   }

   SplitDeviceVector(const SplitVector<T>& vec):_stream(NULL) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice));
      return;
   }

   SplitDeviceVector(const SplitVector<T>& vec,split_gpuStream_t s):_stream(s) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice,s));
      return;
   }

   SplitDeviceVector(const std::vector<T>& vec):_stream(NULL) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice));
      return;
   }

   SplitDeviceVector(const std::vector<T>& vec,split_gpuStream_t s):_stream(s) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice,s));
      return;
   }

   ~SplitDeviceVector() {
      if (_meta == nullptr) {
         return;
      }
      _deallocate(_meta);
   }

   HOSTONLY
   SplitDeviceVector& operator=(const SplitDeviceVector& other) {
      Meta otherMeta = other.getMeta();
      resize(otherMeta.size);
      if (_stream==NULL){
         SPLIT_CHECK_ERR(split_gpuMemcpy(_meta, other._meta, sizeof(Meta) + otherMeta.size * sizeof(T),
                                      split_gpuMemcpyDeviceToDevice));
      }else{
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_meta, other._meta, sizeof(Meta) + otherMeta.size * sizeof(T),
                                      split_gpuMemcpyDeviceToDevice,_stream));
      }
      return *this;
   }

   HOSTONLY
   SplitDeviceVector& operator=(SplitDeviceVector&& other) noexcept {
      if (this == &other) {
         return *this;
      }
      _deallocate(_meta);
      _meta = other._meta;
      _data = other._data;
      other._meta = nullptr;
      other._data = nullptr;
      return *this;
   }

   HOSTONLY
   void* operator new(size_t len) {
      void* _ptr = nullptr;
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&_ptr, len));
      return _ptr;
   }

   HOSTONLY
   void operator delete(void* _ptr) { SPLIT_CHECK_ERR(split_gpuFree(_ptr)); }

   HOSTONLY
   void* operator new[](size_t len) {
      void* _ptr = nullptr;
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&_ptr, len));
      return _ptr;
   }
   HOSTONLY
   void operator delete[](void* _ptr) { SPLIT_CHECK_ERR(split_gpuFree(_ptr)); }

   DEVICEONLY
   size_t device_size() const noexcept { return _meta->size; }

   DEVICEONLY
   size_t device_capacity() const noexcept { return _meta->capacity; }

   HOSTONLY
   size_t size() const noexcept { return (getMeta()).size; }

   HOSTONLY
   size_t capacity() const noexcept { return (getMeta()).capacity; }

   HOSTONLY
   T get(size_t index) const {
      _rangeCheckHost(index);
      return getElementFromDevice(index);
   }

   HOSTONLY
   void set(size_t index, const T& val) {
      _rangeCheckHost(index);
      return setElementFromHost(index, val);
   }

   DEVICEONLY
   T device_get(size_t index) const {
      _rangeCheckDevice(index);
      return _data[index];
   }

   DEVICEONLY
   void device_set(size_t index, const T& val) {
      _rangeCheckDevice(index);
      split::s_atomicExch(&_data[index], val);
      return;
   }

   HOSTDEVICE T* data() noexcept { return &(_data[0]); }

   HOSTDEVICE const T* data() const noexcept { return &(_data[0]); }

   HOSTONLY void reallocate(size_t requested_space) {
      if (requested_space == 0) {
         _deallocate(_meta);
         _meta = nullptr;
         return;
      }

      void* _new_data = _allocate(requested_space);
      const auto currentMeta = getMeta();
      size_t currentSize = currentMeta.size;
      if (_stream==NULL){
         SPLIT_CHECK_ERR(
             split_gpuMemcpy(_new_data, _meta, sizeof(Meta) + currentSize * sizeof(T), split_gpuMemcpyDeviceToDevice));
      }else{
         SPLIT_CHECK_ERR(
             split_gpuMemcpyAsync(_new_data, _meta, sizeof(Meta) + currentSize * sizeof(T), split_gpuMemcpyDeviceToDevice,_stream));
      }
      _deallocate(_meta);
      setupSpace(_new_data);
      auto newMeta = currentMeta;
      newMeta.capacity = requested_space;
      setMeta(newMeta);
      return;
   }

   HOSTONLY
   void clear() noexcept {
      Meta currentMeta=getMeta();
      currentMeta.size=0;
      setMeta(currentMeta);
      return;
   }
   
   DEVICEONLY
   void device_clear() noexcept {
      _meta[MEMBER::SIZE]=0;
      return;
   }

   HOSTONLY
   void reserve(size_t requested_space) {
      Meta currentMeta = getMeta();
      if (requested_space <= currentMeta.capacity) {
         return;
      }
      reallocate(1.5 * requested_space);
      return;
   }

   HOSTONLY
   void resize(size_t newSize) {
      Meta currentMeta = getMeta();
      if (newSize <= currentMeta.size) {
         currentMeta.size = newSize;
         setMeta(currentMeta);
         return;
      }
      reserve(newSize);
      currentMeta = getMeta();
      currentMeta.size = newSize;
      setMeta(currentMeta);
      return;
   }

   HOSTONLY
   void grow() { reserve(capacity() + 1); }

   DEVICEONLY
   void device_resize(size_t newSize) {
      if (newSize>capacity()){
         return;
      }
      _meta[MEMBER::SIZE]=newSize;
   }

   HOSTONLY
   void push_back(const T& val) {
      Meta currentMeta = getMeta();
      resize(currentMeta.size + 1);
      currentMeta.size++;
      setElementFromHost(currentMeta.size - 1, val);
      return;
   }

   DEVICEONLY
   bool device_push_back(const T& val) {
      size_t old = split::s_atomicAdd((unsigned long *)&_meta[MEMBER::SIZE], 1);
      // TODO relpace this > in splitvec as well
      if (old > (_meta->operator[](MEMBER::CAPACITY)) - 1) {
         atomicSub((unsigned *)&_meta[MEMBER::SIZE], 1);
         return false;
      }
      split::s_atomicCAS(&(_data[old]), _data[old], val);
      return true;
   }

   class iterator {

   private:
      const T* _data;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = int64_t;
      using pointer = const T*;
      using reference = const T&;

      iterator(pointer data) : _data(data) {}
      pointer data() const { return _data; }
      pointer operator->() const { return _data; }
      reference operator*() const {
         assert(false);
         return *_data;
      }
      bool operator==(const iterator& other) const { return _data == other._data; }
      bool operator!=(const iterator& other) const { return _data != other._data; }
      iterator& operator++() {
         _data += 1;
         return *this;
      }
      iterator operator++(int) { return iterator(_data + 1); }
      iterator operator--(int) { return iterator(_data - 1); }
      iterator operator--() {
         _data -= 1;
         return *this;
      }
      iterator& operator+=(int64_t offset) {
         _data += offset;
         return *this;
      }
      iterator& operator-=(int64_t offset) {
         _data -= offset;
         return *this;
      }
      iterator operator+(int64_t offset) const {
         iterator itt(*this);
         return itt += offset;
      }
      iterator operator-(int64_t offset) const {
         iterator itt(*this);
         return itt -= offset;
      }
   };

   // Device Iterators
   class device_iterator {

   private:
      T* _data;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = int64_t;
      using pointer = T*;
      using reference = T&;

      // device_iterator(){}
      DEVICEONLY
      device_iterator(pointer data) : _data(data) {}

      DEVICEONLY
      pointer data() { return _data; }
      DEVICEONLY
      pointer operator->() { return _data; }
      DEVICEONLY
      reference operator*() { return *_data; }

      DEVICEONLY
      bool operator==(const device_iterator& other) const { return _data == other._data; }
      DEVICEONLY
      bool operator!=(const device_iterator& other) const { return _data != other._data; }
      DEVICEONLY
      device_iterator& operator++() {
         _data += 1;
         return *this;
      }
      DEVICEONLY
      device_iterator operator++(int) { return device_iterator(_data + 1); }
      DEVICEONLY
      device_iterator operator--(int) { return device_iterator(_data - 1); }
      DEVICEONLY
      device_iterator operator--() {
         _data -= 1;
         return *this;
      }
      DEVICEONLY
      device_iterator& operator+=(int64_t offset) {
         _data += offset;
         return *this;
      }
      DEVICEONLY
      device_iterator& operator-=(int64_t offset) {
         _data -= offset;
         return *this;
      }
      DEVICEONLY
      device_iterator operator+(int64_t offset) const {
         device_iterator itt(*this);
         return itt += offset;
      }
      DEVICEONLY
      device_iterator operator-(int64_t offset) const {
         device_iterator itt(*this);
         return itt -= offset;
      }
   };

   class const_device_iterator {

   private:
      const T* _data;

   public:
      using device_iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = int64_t;
      using pointer = const T*;
      using reference = const T&;

      DEVICEONLY
      const_device_iterator(pointer data) : _data(data) {}

      DEVICEONLY
      pointer data() const { return _data; }
      DEVICEONLY
      pointer operator->() const { return _data; }
      DEVICEONLY
      reference operator*() const { return *_data; }

      DEVICEONLY
      bool operator==(const const_device_iterator& other) const { return _data == other._data; }
      DEVICEONLY
      bool operator!=(const const_device_iterator& other) const { return _data != other._data; }
      DEVICEONLY
      const_device_iterator& operator++() {
         _data += 1;
         return *this;
      }
      DEVICEONLY
      const_device_iterator operator++(int) { return const_iterator(_data + 1); }
      DEVICEONLY
      const_device_iterator operator--(int) { return const_iterator(_data - 1); }
      DEVICEONLY
      const_device_iterator operator--() {
         _data -= 1;
         return *this;
      }
      DEVICEONLY
      const_device_iterator& operator+=(int64_t offset) {
         _data += offset;
         return *this;
      }
      DEVICEONLY
      const_device_iterator& operator-=(int64_t offset) {
         _data -= offset;
         return *this;
      }
      DEVICEONLY
      const_device_iterator operator+(int64_t offset) const {
         const_device_iterator itt(*this);
         return itt += offset;
      }
      DEVICEONLY
      const_device_iterator operator-(int64_t offset) const {
         const_device_iterator itt(*this);
         return itt -= offset;
      }
   };

   HOSTONLY
   iterator begin() const noexcept { return iterator(_data); }

   HOSTONLY
   iterator end() const noexcept { return iterator(_data + size()); }

   DEVICEONLY
   device_iterator device_begin() noexcept { return device_iterator(_data); }

   DEVICEONLY
   const_device_iterator device_begin() const noexcept { return const_device_iterator(_data); }

   DEVICEONLY
   device_iterator device_end() noexcept { return device_iterator(_data + device_size()); }

   DEVICEONLY
   const_device_iterator device_end() const noexcept { return const_device_iterator(_data + device_size()); }

   HOSTONLY
   T back() const noexcept { return get(size()-1); }

   HOSTONLY
   T front() const noexcept { return get(0); }

   DEVICEONLY
   T& device_back() noexcept { return _data[size() - 1]; }

   DEVICEONLY
   T& device_front() noexcept { return _data[0]; }

   DEVICEONLY
   const T& device_back() const noexcept { return _data[size() - 1]; }

   DEVICEONLY
   const T& device_front() const noexcept { return _data[0]; }

   HOSTONLY
   void set(const iterator& it, T val ){
      size_t index = it.data()-_data;
      return setElementFromHost(index, val);
   }

   T get(const iterator& it){
      size_t index = it.data()-_data;
      return getElementFromDevice(index);
   }

   HOSTONLY
   void remove_from_back(size_t n) noexcept {
      const size_t end = size() - n;
      Meta currentMeta=getMeta();
      currentMeta.size=end;
      setMeta(currentMeta);
   }

   DEVICEONLY
   void device_remove_from_back(size_t n) noexcept {
      const size_t end = device_size() - n;
      _meta[MEMBER::SIZE]=end;
   }

   HOSTONLY
   void pop_back() noexcept {
      remove_from_back(1);
   }

   DEVICEONLY
   void device_pop_back() noexcept {
      device_remove_from_back(1);
   }

   HOSTONLY
   iterator erase(iterator it) noexcept {
      const int64_t index = it.data() - begin().data();
      Meta currentMeta=getMeta();
      for (auto i = index; i < size() - 1; i++) {
         set(i,get(i+1));
      }
      currentMeta.size-=1;
      setMeta(currentMeta);
      iterator retval = &_data[index];
      return retval;
   }

   HOSTONLY
   iterator erase(iterator p0, iterator p1) noexcept {
      const int64_t start = p0.data() - begin().data();
      const int64_t end = p1.data() - begin().data();
      const int64_t offset = end - start;
      Meta currentMeta=getMeta();
      for (auto i = start; i < size() - offset; ++i) {
         set(i,get(i+offset));
      }
      currentMeta.size-=end - start;
      setMeta(currentMeta);
      iterator it = &_data[start];
      return it;
   }

   HOSTONLY
   iterator insert(iterator it, const T& val) {

      // If empty or inserting at the end no relocating is needed
      if (it == end()) {
         push_back(val);
         return end()--;
      }

      int64_t index = it.data() - begin().data();
      if (index < 0 || index > size()) {
         throw std::out_of_range("Insert");
      }

      // Do we do need to increase our capacity?
      if (size() == capacity()) {
         grow();
      }

      for (int64_t i = size() - 1; i >= index; i--) {
         set(i+1,get(i));
      }

      set(index,val);
      Meta currentMeta = getMeta();
      currentMeta.size++;
      setMeta(currentMeta);
      return iterator(_data + index);
   }


}; // SplitDeviceVector

/*Equal operator*/
template <typename T>
static inline HOSTONLY bool operator==(const SplitDeviceVector<T>& lhs, const SplitDeviceVector<T>& rhs) noexcept {
   if (lhs.size() != rhs.size()) {
      return false;
   }
   for (size_t i = 0; i < lhs.size(); i++) {
      if (!(lhs.get(i) == rhs.get(i))) {
         return false;
      }
   }
   // if we end up here the vectors are equal
   return true;
}

/*Not-Equal operator*/
template <typename T>
static inline HOSTONLY bool operator!=(const SplitDeviceVector<T>& lhs, const SplitDeviceVector<T>& rhs) noexcept {
   return !(rhs == lhs);
}
#endif

} // namespace split

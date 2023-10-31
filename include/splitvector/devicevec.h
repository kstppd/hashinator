#pragma once
#include "archMacros.h"
#include "gpu_wrappers.h"
#include <cassert>
#ifdef __NVCC__
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#define HOSTONLY __host__
#define DEVICEONLY __device__
#define HOSTDEVICE __host__ __device__

namespace split {

template <typename T>
class DeviceVector {
   static_assert(std::is_trivially_copyable<T>::value && "DeviceVector only works for POD types");

private:
   enum MEMBER { SIZE, CAPACITY };

   // packed to 1 cache line
   struct __attribute__((__packed__)) Meta {
      size_t size;
      size_t capacity;
      char padding[64 - 2 * sizeof(size_t)]; // pad up to cache line
      HOSTDEVICE
      inline size_t& operator[](MEMBER member) noexcept {
         switch (member) {
         case MEMBER::SIZE:
            return *reinterpret_cast<size_t*>(this);
         case MEMBER::CAPACITY:
            return *(reinterpret_cast<size_t*>(this) + 1);
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
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMalloc((void**)&_ptr, sizeof(Meta) + sz * sizeof(T)));
      } else {
         SPLIT_CHECK_ERR(split_gpuMallocAsync((void**)&_ptr, sizeof(Meta) + sz * sizeof(T), _stream));
      }
      return _ptr;
   }

   void _deallocate(void* _ptr) {
      if (_ptr == nullptr) {
         return;
      }
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuFree((void*)_ptr));
      } else {
         SPLIT_CHECK_ERR(split_gpuFreeAsync((void*)_ptr, _stream));
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
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost, _stream));
      }
      return buffer;
   }

   HOSTONLY
   inline void getMeta(Meta& buffer) const noexcept {
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&buffer, _meta, sizeof(Meta), split_gpuMemcpyDeviceToHost, _stream));
      }
   }

   HOSTONLY
   inline void setMeta(const Meta& buffer) noexcept {
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(_meta, &buffer, sizeof(Meta), split_gpuMemcpyHostToDevice));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_meta, &buffer, sizeof(Meta), split_gpuMemcpyHostToDevice, _stream));
      }
   }

   HOSTONLY
   inline T getElementFromDevice(const size_t index) const noexcept {
      T retval;
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(&retval, &_data[index], sizeof(T), split_gpuMemcpyDeviceToHost));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&retval, &_data[index], sizeof(T), split_gpuMemcpyDeviceToHost, _stream));
      }
      return retval;
   }

   HOSTONLY
   inline void setElementFromHost(const size_t index, const T& val) noexcept {
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(&_data[index], &val, sizeof(T), split_gpuMemcpyHostToDevice));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&_data[index], &val, sizeof(T), split_gpuMemcpyHostToDevice, _stream));
      }
      return;
   }

public:
   DeviceVector(size_t sz = 0) : _stream(NULL) {
      void* ptr = _allocate(sz);
      setupSpace(ptr);
      size_t s = sz;
      SPLIT_CHECK_ERR(split_gpuMemcpy(&_meta->size, &s, sizeof(size_t), split_gpuMemcpyHostToDevice));
      SPLIT_CHECK_ERR(split_gpuMemcpy(&_meta->capacity, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice));
   }

   DeviceVector(size_t sz, split_gpuStream_t s) : _stream(s) {
      void* ptr = _allocate(sz);
      setupSpace(ptr);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&_meta->size, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice, s));
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(&_meta->capacity, &sz, sizeof(size_t), split_gpuMemcpyHostToDevice, s));
   }

   DeviceVector(const DeviceVector<T>& other) : _stream(NULL) {
      Meta otherMeta = other.getMeta();
      void* ptr = _allocate(otherMeta.size);
      setupSpace(ptr);
      Meta newMeta{.size = otherMeta.size, .capacity = otherMeta.size};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, other._data, otherMeta.size * sizeof(T), split_gpuMemcpyDeviceToDevice));
      return;
   }

   DeviceVector(const DeviceVector<T>& other, split_gpuStream_t s) : _stream(s) {
      Meta otherMeta = other.getMeta();
      void* ptr = _allocate(otherMeta.size);
      setupSpace(ptr);
      Meta newMeta{.size = otherMeta.size, .capacity = otherMeta.size};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(
          split_gpuMemcpyAsync(_data, other._data, otherMeta.size * sizeof(T), split_gpuMemcpyDeviceToDevice, s));
      return;
   }

   DeviceVector(const DeviceVector<T>&& other) : _stream(NULL) {
      _meta = other._meta;
      _data = other._data;
      other._meta = nullptr;
      other._data = nullptr;
   }

   DeviceVector(const DeviceVector<T>&& other, split_gpuStream_t s) : _stream(s) {
      _meta = other._meta;
      _data = other._data;
      other._meta = nullptr;
      other._data = nullptr;
   }

   DeviceVector(const SplitVector<T>& vec) : _stream(NULL) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice));
      return;
   }

   DeviceVector(const SplitVector<T>& vec, split_gpuStream_t s) : _stream(s) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice, s));
      return;
   }

   DeviceVector(const std::vector<T>& vec) : _stream(NULL) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpy(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice));
      return;
   }

   DeviceVector(const std::vector<T>& vec, split_gpuStream_t s) : _stream(s) {
      void* ptr = _allocate(vec.size());
      setupSpace(ptr);
      Meta newMeta{.size = vec.size(), .capacity = vec.size()};
      setMeta(newMeta);
      SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_data, vec.data(), vec.size() * sizeof(T), split_gpuMemcpyHostToDevice, s));
      return;
   }

   ~DeviceVector() {
      if (_meta == nullptr) {
         return;
      }
      _deallocate(_meta);
   }

   HOSTONLY
   DeviceVector& operator=(const DeviceVector& other) {
      Meta otherMeta = other.getMeta();
      resize(otherMeta.size);
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(split_gpuMemcpy(_meta, other._meta, sizeof(Meta) + otherMeta.size * sizeof(T),
                                         split_gpuMemcpyDeviceToDevice));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_meta, other._meta, sizeof(Meta) + otherMeta.size * sizeof(T),
                                              split_gpuMemcpyDeviceToDevice, _stream));
      }
      return *this;
   }

   HOSTONLY
   DeviceVector& operator=(DeviceVector&& other) noexcept {
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

   DEVICEONLY T& at(size_t index) {
      _rangeCheckDevice(index);
      return _data[index];
   }

   DEVICEONLY const T& at(size_t index) const {
      _rangeCheckDevice(index);
      return _data[index];
   }

   DEVICEONLY T& operator[](size_t index) noexcept { return _data[index]; }

   DEVICEONLY const T& operator[](size_t index) const noexcept { return _data[index]; }

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
      if (_stream == NULL) {
         SPLIT_CHECK_ERR(
             split_gpuMemcpy(_new_data, _meta, sizeof(Meta) + currentSize * sizeof(T), split_gpuMemcpyDeviceToDevice));
      } else {
         SPLIT_CHECK_ERR(split_gpuMemcpyAsync(_new_data, _meta, sizeof(Meta) + currentSize * sizeof(T),
                                              split_gpuMemcpyDeviceToDevice, _stream));
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
      Meta currentMeta = getMeta();
      currentMeta.size = 0;
      setMeta(currentMeta);
      return;
   }

   DEVICEONLY
   void device_clear() noexcept {
      _meta[MEMBER::SIZE] = 0;
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
      if (newSize > capacity()) {
         return;
      }
      _meta[MEMBER::SIZE] = newSize;
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
      size_t old = split::s_atomicAdd((unsigned long*)&_meta[MEMBER::SIZE], 1);
      // TODO relpace this > in splitvec as well
      if (old > (_meta->operator[](MEMBER::CAPACITY)) - 1) {
         atomicSub((unsigned*)&_meta[MEMBER::SIZE], 1);
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
   T back() const noexcept { return get(size() - 1); }

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
   void set(const iterator& it, T val) {
      size_t index = it.data() - _data;
      return setElementFromHost(index, val);
   }

   T get(const iterator& it) {
      size_t index = it.data() - _data;
      return getElementFromDevice(index);
   }

   HOSTONLY
   void remove_from_back(size_t n) noexcept {
      const size_t end = size() - n;
      Meta currentMeta = getMeta();
      currentMeta.size = end;
      setMeta(currentMeta);
   }

   DEVICEONLY
   void device_remove_from_back(size_t n) noexcept {
      const size_t end = device_size() - n;
      _meta[MEMBER::SIZE] = end;
   }

   HOSTONLY
   void pop_back() noexcept { remove_from_back(1); }

   DEVICEONLY
   void device_pop_back() noexcept { device_remove_from_back(1); }

   HOSTONLY
   iterator erase(iterator it) noexcept {
      const int64_t index = it.data() - begin().data();
      Meta currentMeta = getMeta();
      for (auto i = index; i < size() - 1; i++) {
         set(i, get(i + 1));
      }
      currentMeta.size -= 1;
      setMeta(currentMeta);
      iterator retval = &_data[index];
      return retval;
   }

   HOSTONLY
   iterator erase(iterator p0, iterator p1) noexcept {
      const int64_t start = p0.data() - begin().data();
      const int64_t end = p1.data() - begin().data();
      const int64_t offset = end - start;
      Meta currentMeta = getMeta();
      for (auto i = start; i < size() - offset; ++i) {
         set(i, get(i + offset));
      }
      currentMeta.size -= end - start;
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
         set(i + 1, get(i));
      }

      set(index, val);
      Meta currentMeta = getMeta();
      currentMeta.size++;
      setMeta(currentMeta);
      return iterator(_data + index);
   }

   template <typename InputIterator, class = typename std::enable_if<!std::is_integral<InputIterator>::value>::type>
   HOSTONLY iterator insert(iterator it, InputIterator p0, InputIterator p1) {

      const int64_t count = std::distance(p0, p1);
      const int64_t index = it.data() - begin().data();

      if (index < 0 || index > size()) {
         throw std::out_of_range("Insert");
      }

      size_t old_size = size();
      resize(size() + count);

      iterator retval = &_data[index];

      // Copy
      for (int64_t i = old_size - 1; i >= index; i--) {
         set(count + i, get(i));
      }

      // Overwrite
      size_t i = index;
      for (auto p = p0; p != p1; ++p) {
         set(i, get(p));
         i++;
      }
      return retval;
   }

}; // DeviceVector

/*Equal operator*/
template <typename T>
static inline HOSTONLY bool operator==(const DeviceVector<T>& lhs, const DeviceVector<T>& rhs) noexcept {
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
static inline HOSTONLY bool operator!=(const DeviceVector<T>& lhs, const DeviceVector<T>& rhs) noexcept {
   return !(rhs == lhs);
}
} // namespace split

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
#include <iostream>
#include <cassert>
#include <cstring>
#include <stdlib.h>
#include <algorithm>
#include <memory>
#include "split_allocators.h"


#ifndef SPLIT_HOST_ONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#define HOSTONLY __host__
#define DEVICEONLY __device__
#define HOSTDEVICE __host__ __device__
template <typename T>
using DefaultAllocator = split::split_unified_allocator<T>;
template <typename T>
using DefaultMetaAllocator = split::split_unified_allocator<size_t>;

#else

#define HOSTONLY
#define DEVICEONLY
#define HOSTDEVICE
template <typename T>
using DefaultAllocator = split::split_host_allocator<T>;
template <typename T>
using DefaultMetaAllocator = split::split_host_allocator<size_t>;

#endif

namespace split{


   template <typename T> void swap(T& t1, T& t2) {
       T tmp = std::move(t1);
       t1 = std::move(t2);
       t2 = std::move(tmp);
   }


   template<typename T,
            class Allocator=DefaultAllocator<T>,
            class Meta_Allocator=DefaultMetaAllocator<size_t>>
   class SplitVector{
      
      private:
         T* _data=nullptr;                  // actual pointer to our data      
         size_t* _size;                     // number of elements in vector.
         size_t* _capacity;                 // number of allocated elements
         size_t  _alloc_multiplier = 2;     // host variable; multiplier for  when reserving more space
         Allocator _allocator;              // Allocator used to allocate and deallocate memory;
         Meta_Allocator _meta_allocator;    // Allocator used to allocate and deallocate memory for metadata
                                            //   (currently: _size, _capacity);
 
         void _check_ptr(void* ptr){
            if (ptr==nullptr){
               throw std::bad_alloc();
            }
         }

         /*Internal range check for use in .at()*/
         HOSTDEVICE void _rangeCheck(size_t index){
            if (index>=size()){printf("Tried indexing %d/%d\n",(int)index,(int)size());}
            assert(index<size() &&  "out of range ");
         }

         /*Allocation/Deallocation only on host*/
         HOSTONLY void _allocate(size_t size){
            _size=_allocate_and_construct(size);
            _capacity=_allocate_and_construct(size);
            _check_ptr(_size);
            _check_ptr(_capacity);
            if (size==0){return;}
            _data=_allocate_and_construct(size,T());
            _check_ptr(_data);
            if (_data == nullptr){
               _deallocate();
               throw std::bad_alloc();
            }
         }

         HOSTONLY void _deallocate(){
               if (_data!=nullptr){
                  _deallocate_and_destroy(capacity(),_data);
                  _data=nullptr;
               }
               _deallocate_and_destroy(_capacity);
               _deallocate_and_destroy(_size);
         }

         HOSTONLY T* _allocate_and_construct(size_t n, const T &val){
            T* _ptr=_allocator.allocate(n);
            for (size_t i=0; i < n; i++){
               _allocator.construct(&_ptr[i],val);
            }
            return _ptr;
         }
         HOSTONLY size_t* _allocate_and_construct(const size_t &val){
            size_t* _ptr=_meta_allocator.allocate(1);
            _meta_allocator.construct(_ptr,val);
            return _ptr;
         }

         HOSTONLY void _deallocate_and_destroy(size_t n,T* _ptr){
            for (size_t i=0; i < n; i++){
               _allocator.destroy(&_ptr[i]);
            }
            _allocator.deallocate(_ptr,n);
         }

         HOSTONLY void _deallocate_and_destroy(size_t* ptr){
            _meta_allocator.deallocate(ptr,1);
         }

      public:
         /* Available Constructors :
          *    -- SplitVector()                       --> Default constructor. Almost a no OP but _size and _capacity have  to be allocated for device usage. 
          *    -- SplitVector(size_t)                 --> Instantiates a splitvector with a specific size. (capacity == size)
          *    -- SplitVector(size_t,T)               --> Instantiates a splitvector with a specific size and sets all elements to T.(capacity == size)
          *    -- SplitVector(SplitVector&)           --> Copy constructor. 
          *    -- SplitVector(SplitVector&&)          --> Move constructor. 
          *    -- SplitVector(std::initializer_list&) --> Creates a SplitVector and copies over the elemets of the init. list. 
          *    -- SplitVector(std::vector&)           --> Creates a SplitVector and copies over the elemets of the std vector
          * */

         /*Constructors*/
         HOSTONLY explicit   SplitVector(){
            this->_allocate(0); //seems counter-intuitive based on stl but it is not!
         }

         HOSTONLY explicit   SplitVector(size_t size){
               this->_allocate(size);
         }

         HOSTONLY explicit  SplitVector(size_t size, const T &val){
               this->_allocate(size);
               for (size_t i=0; i<size; i++){
                  _data[i]=val;
               }
            }

         HOSTONLY explicit SplitVector(const SplitVector<T,Allocator,Meta_Allocator> &other){
               const size_t size_to_allocate = other.size();
               this->_allocate(size_to_allocate);
               for (size_t i=0; i<size_to_allocate; i++){
                  _data[i]=other._data[i];
               }
            }
         
         HOSTONLY SplitVector(SplitVector<T,Allocator,Meta_Allocator> &&other)noexcept{
               _data=other._data;
               *_size=other.size();
               *_capacity=other.capacity();
               *(other._capacity)=0;
               *(other._size)=0;
               other._data=nullptr;
            }

         HOSTONLY explicit SplitVector(std::initializer_list<T> init_list){
               this->_allocate(init_list.size());
               for (size_t i =0 ; i< size();i++){
                  _data[i]=init_list.begin()[i];
               }
            }
    
         HOSTONLY explicit  SplitVector(const std::vector<T> &other ){
               this->_allocate(other.size());
               for (size_t i=0; i<size(); i++){
                  _data[i]=other[i];
               }
            }

         //Destructor
         HOSTONLY ~SplitVector(){
            _deallocate();
         }

         
         /*Custom Assignment operator*/
         HOSTONLY  SplitVector<T,Allocator,Meta_Allocator>& operator=(const SplitVector<T,Allocator,Meta_Allocator>& other){
            //Match other's size prior to copying
            resize(other.size());
            for (size_t i=0; i< other.size(); i++){
               _data[i]=other._data[i];
            }
            return *this;
         }

         HOSTONLY  SplitVector<T,Allocator,Meta_Allocator>& operator=(SplitVector<T,Allocator,Meta_Allocator>&& other)noexcept{
            if (this==&other){
               return *this;
            }

            _deallocate_and_destroy(capacity(),_data);
            _data=other._data;
            *_size=other.size();
            *_capacity=other.capacity();
            *(other._capacity)=0;
            *(other._size)=0;
            other._data=nullptr;
            return *this;
         }

         HOSTONLY
         void *operator new(size_t len) {
            void *ptr=Allocator::allocate_raw(len);
            return ptr;
         }

         HOSTONLY
         void operator delete(void *ptr) {
            Allocator::deallocate(ptr,1);
         }

         HOSTONLY
         void* operator new[] (size_t len) {
            void *ptr=Allocator::allocate_raw(len);
            return ptr;
         }

         HOSTONLY
         void operator delete[] (void* ptr) {
            Allocator::deallocate(ptr);
         }

         #ifndef SPLIT_HOST_ONLY
         //Method that return a pointer which can be passed to GPU kernels
         //Has to be cudaFree'd after use otherwise memleak (small one but still)!
         HOSTONLY
         SplitVector<T,Allocator,Meta_Allocator>* upload(cudaStream_t stream = 0 )noexcept{
            SplitVector* d_vec;
            optimizeGPU(stream);
            cudaMalloc((void **)&d_vec, sizeof(SplitVector));
            cudaMemcpyAsync(d_vec, this, sizeof(SplitVector),cudaMemcpyHostToDevice,stream);
            return d_vec;
         }

         /*Manually prefetch data on Device*/
         HOSTONLY void optimizeGPU(cudaStream_t stream = 0)noexcept{
            int device;
            cudaGetDevice(&device);
            CheckErrors("Prefetch GPU-Device-ID");
            cudaMemPrefetchAsync(_data ,capacity()*sizeof(T),device,stream);
            CheckErrors("Prefetch GPU");
         }

         /*Manually prefetch data on Host*/
         HOSTONLY void optimizeCPU(cudaStream_t stream = 0)noexcept{
            cudaMemPrefetchAsync(_data ,capacity()*sizeof(T),cudaCpuDeviceId,stream);
            CheckErrors("Prefetch CPU");
         }

         HOSTONLY void streamAttach(cudaStream_t s,uint32_t flags=cudaMemAttachSingle){
            cudaStreamAttachMemAsync( s,(void*)_size,     sizeof(size_t),flags );
            cudaStreamAttachMemAsync( s,(void*)_capacity, sizeof(size_t),flags );
            cudaStreamAttachMemAsync( s,(void*)_data,     *_capacity*sizeof(T), flags );
            CheckErrors("Stream Attach");
            return;
         }
         #endif

         /* Custom swap mehtod. 
          * Pointers outside of splitvector's source
          * are invalidated after swap is called.
          */
         void swap(SplitVector<T,Allocator,Meta_Allocator>& other) noexcept{
            if (*this==other){ //no need to do any work
               return;
            }
            split::swap(_data,other._data);
            split::swap(_size,other._size);
            split::swap(_capacity,other._capacity);
            split::swap(_allocator,other._allocator);
            return;
         }



         /************STL compatibility***************/
         /*Returns number of elements in this container*/
         HOSTDEVICE const size_t& size() const noexcept{
            return *_size;
         }

         /*Bracket accessor - no bounds check*/
         HOSTDEVICE T& operator [](size_t index)noexcept{
               return _data[index];
         } 
                  
         /*Const Bracket accessor - no bounds check*/
         HOSTDEVICE const T& operator [](size_t index)const noexcept{
               return _data[index];
         } 

         /*at accesor with bounds check*/
         HOSTDEVICE T& at(size_t index){
            _rangeCheck(index);
            return _data[index];
         }
         
         /*const at accesor with bounds check*/
         HOSTDEVICE const T& at(size_t index)const{
            _rangeCheck(index);
            return _data[index];
         }

         /*Return a raw pointer to our data similar to stl vector*/
         HOSTDEVICE T* data()noexcept{
            return &(_data[0]);
         }

         /*Return a raw pointer to our data similar to stl vector*/
         HOSTDEVICE const T* data() const noexcept{
            return &(_data[0]);
         }

         /* Size Modifiers*/

         /*Reallocates data to a bigeer chunk of memory. At some point
          * this should be udpated to use move semantics*/
         HOSTONLY void reallocate(size_t requested_space){
            T* _new_data;
            _new_data=_allocate_and_construct(requested_space,T());
            if (_new_data==nullptr){
               _deallocate_and_destroy(requested_space,_new_data);
               this->_deallocate();
               throw std::bad_alloc();
            }
            
            //Copy over
            for (size_t i=0; i<size();i++){
               _new_data[i] = _data[i];
            }

            //Deallocate old space
            _deallocate_and_destroy(capacity(),_data);

            //Swap pointers & update capacity
            //Size remains the same ofc
            _data=_new_data;
            *_capacity=requested_space;
            return ;

         }

         /*Reserve method:
          *Supports only host reserving.
          *Will never reduce the vector's size.
          *Memory location will change so any old pointers/iterators
          *will be invalidated after a call.
          */
         HOSTONLY
         void reserve(size_t requested_space,bool eco=false){
            size_t current_space=*_capacity;
            //Vector was default initialized
            if (_data==nullptr){
               _deallocate();
               _allocate(requested_space);
               *_size=0;
               return;
            }
            //Nope.
            if (requested_space <= current_space){
               for (size_t i=size(); i<requested_space; ++i ){
                  _allocator.construct(&_data[i],T());
               }
               return ;
            }
            //If the users passes eco=true we allocate
            //exactly what was requested
            if (!eco){
               requested_space*=_alloc_multiplier;
            }
            reallocate(requested_space);
            return;
         }

          /* 
            Resize method:
            Supports only host resizing.
            If new size is smaller than the current size we just reduce size but 
            the capacity remains the same
            Memory location will change so any old pointers/iterators
            will be invalid from now on.
         */
         HOSTONLY
         void resize(size_t newSize,bool eco=false ){
            //Let's reserve some space and change our size
            if (newSize<=size()){
               *_size=newSize; 
               return;
            }
            reserve(newSize,eco);
            *_size  =newSize; 
         }
         
         #ifndef SPLIT_HOST_ONLY
         DEVICEONLY
         void device_resize(size_t newSize){
            if (newSize>capacity()){
               assert(0 && "Splitvector has a catastrophic failure trying to resize on device.");
            }
            for (size_t i=size(); i<newSize; ++i ){
               _allocator.construct(&_data[i],T());
            }
            *_size=newSize; 
         }
         #endif


         HOSTONLY
         void grow(){
            reserve(capacity()+1);
         }
 
         HOSTONLY
         void shrink_to_fit(){
            size_t curr_cap =*_capacity;
            size_t curr_size=*_size;

            if (curr_cap == curr_size){
               return;
            }

            reallocate(curr_size);
            return;
         }
         
         /*Removes the last element of the vector*/
         HOSTDEVICE
         void pop_back(){
            if (size()>0){
               remove_from_back(1);
            }
            return;
         }

         //Removes n elements from the back of the vector
         //and properly handles object destruction
         HOSTDEVICE
        void remove_from_back(size_t n)noexcept{
          const size_t end = size() - n;
          for (auto i = size(); i > end;) {
            (_data + --i)->~T();
          }
          *_size = end;
        }

         HOSTDEVICE
         void clear()noexcept{
             for (size_t i = 0; i < size();i++) {
               _data[i].~T();
             }
            *_size=0;
            return;
         }

         HOSTDEVICE
         size_t capacity() const noexcept{
            return *_capacity;
         }

         HOSTDEVICE
         T& back()noexcept{ return _data[*_size-1]; }

         HOSTDEVICE
         const T& back()const noexcept{return _data[*_size-1];}
         
         HOSTDEVICE
         T& front()noexcept{return _data[0];}
         
         HOSTDEVICE
         const T& front() const noexcept{ return _data[0]; }

         HOSTDEVICE 
         bool empty() const noexcept{
            return  size()==0;
         }

         /* 
            PushBack  method:
            Supports only host  resizing.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalid from now on.
            Not thread safe
         */      
         HOSTONLY 
         void push_back(const T& val){
            // If we have no allocated memory because the default ctor was used then 
            // allocate one element, set it and return 
            if (_data==nullptr){
               *this=SplitVector<T,Allocator,Meta_Allocator>(1,val);
               return;
            }
            resize(size()+1);
            _data[size()-1] = val;
            return;
         }
         

         HOSTONLY  
         void push_back(const T&& val){

            // If we have no allocated memory because the default ctor was used then 
            // allocate one element, set it and return 
            if (_data==nullptr){
               *this=SplitVector<T,Allocator,Meta_Allocator>(1,std::move(val));
               return;
            }
            resize(size()+1);
            _data[size()-1] = std::move(val);
            return;
         }

         #ifndef SPLIT_HOST_ONLY
         /* 
            Device PushBack  method:
            Will never reduce the vector's size.
            Memory location will not change.
            Will crash if it runs out of space
         */      
         DEVICEONLY 
         void device_push_back(const T& val){
            size_t old= atomicAdd((unsigned int*)_size, 1);
            if (old>=capacity()){
               assert(0 && "Splitvector has a catastrophic failure trying to pushback on device because the vector has no space available.");
            }
            atomicCAS(&(_data[old]), _data[old],val);
            return;
         }
         

         DEVICEONLY 
         void device_push_back(const T&& val){

            //We need at least capacity=size+1 otherwise this 
            //pushback cannot be done
            size_t old= atomicAdd((unsigned int*)_size, 1);
            if (old>=capacity()){
               assert(0 && "Splitvector has a catastrophic failure trying to pushback on device because the vector has no space available.");
            }
            atomicCAS(&(_data[old]), _data[old],std::move(val));
            return;
         }
         #endif

         //Iterators
         class iterator{
             
            private:
            T* _data;
            
            public:
            
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = int64_t;
            using pointer = T*;
            using reference = T&;

            //iterator(){}
            HOSTDEVICE
            iterator(pointer data) : _data(data) {}

            HOSTDEVICE
            pointer data() { return _data; }
            HOSTDEVICE
            pointer operator->() { return _data; }
            HOSTDEVICE
            reference operator*() { return *_data; }

            HOSTDEVICE
            bool operator==(const iterator& other)const{
              return _data == other._data;
            }
            HOSTDEVICE
            bool operator!=(const iterator& other)const {
              return _data != other._data;
            }
            HOSTDEVICE
            iterator& operator++(){
              _data += 1;
              return *this;
            }
            HOSTDEVICE
            iterator operator++(int){
              return iterator(_data + 1);
            }
            HOSTDEVICE
            iterator operator--(int){
              return iterator(_data - 1);
            }
            HOSTDEVICE
            iterator operator--(){
              _data-=1;
              return *this;
            }
         };

         class const_iterator{
             
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
            pointer data()const { return _data; }
            HOSTDEVICE
            pointer operator->()const  { return _data; }
            HOSTDEVICE
            reference operator*()const  { return *_data; }

            HOSTDEVICE
            bool operator==(const const_iterator& other)const{
              return _data == other._data;
            }
            HOSTDEVICE
            bool operator!=(const const_iterator& other)const {
              return _data != other._data;
            }
            HOSTDEVICE
            const_iterator& operator++(){
              _data += 1;
              return *this;
            }
            HOSTDEVICE
            const_iterator operator++(int){
              return const_iterator(_data + 1);
            }
            HOSTDEVICE
            const_iterator operator--(int){
              return const_iterator(_data - 1);
            }
            HOSTDEVICE
            const_iterator operator--(){
              _data-=1;
              return *this;
            }
         };
         
         HOSTDEVICE
         iterator begin() noexcept{
            return iterator(_data);
         }

         HOSTDEVICE
         const_iterator begin()const noexcept{
            return const_iterator(_data);
         }

         HOSTDEVICE
         iterator end() noexcept{
            return iterator(_data+size());
         }
         HOSTDEVICE
         const_iterator end() const noexcept{
            return const_iterator(_data+size());
         }

         HOSTONLY
         iterator insert (iterator it, const T& val){
            
            //If empty or inserting at the end no relocating is needed
            if (it==end()){
               push_back(val);
               return end()--;
            }

            int64_t index=it.data()-begin().data();
            if (index<0 || index>size()){
               throw std::out_of_range("Insert");
            }
            
            //Do we do need to increase our capacity?
            if (size()==capacity()){
               grow();
            }

            for(int64_t  i = size() - 1; i >= index; i--){
               _data[i+1] = _data[i];
            }
            _data[index] = val;
            *_size=*_size+1;
            return iterator(_data+index);
         }

         HOSTONLY
         iterator insert(iterator it,const size_t elements, const T& val){

            int64_t index=it.data()-begin().data();
            size_t oldsize=size();
            size_t newSize=oldsize+elements;
            if (index<0 || index>size()){
               throw std::out_of_range("Insert");
            }
            
            //Do we do need to increase our capacity?
            if (newSize>size()){
               resize(newSize);
            }

            it = begin().data() + index;
            iterator last= it.data()+oldsize;
            std::copy_backward(it,last,last.data()+elements);
            last= it.data()+elements;
            std::fill(it,last,val);
            iterator retval = &_data[index];
            return retval;
         }

            
         template<typename InputIterator, class = typename std::enable_if< !std::is_integral<InputIterator>::value >::type>
         HOSTONLY 
         iterator insert(iterator it, InputIterator p0, InputIterator p1){

            const int64_t count = std::distance(p0, p1);
            const int64_t index = it.data() - begin().data();
      
            if (index<0 || index>size()){
               throw std::out_of_range("Insert");
            }

            size_t old_size = size();
            if (size() + count > capacity()) {
               resize(capacity() + count);
            }

            iterator retval = &_data[index];
            std::move(retval, iterator(&_data[old_size]), retval.data() + count);
            std::copy(p0, p1, retval);
            return retval;
         }

         #ifndef SPLIT_HOST_ONLY 
         template<typename InputIterator, class = typename std::enable_if< !std::is_integral<InputIterator>::value >::type>
         DEVICEONLY 
         iterator device_insert(iterator it, InputIterator p0, InputIterator p1) noexcept{

            const int64_t count = p1.data()-p0.data();
            const int64_t index = it.data() - begin().data();
      
            if (index<0 || index>size()){
               assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no space available.");
            }

            if (size() + count > capacity()) {
               assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no space available.");
            }

            //Increase size;
            device_resize(size()+count);
            for (size_t i=0; i<count; ++i){
               _data[index+i] = *(p0.data()+i);
            }
            iterator retval = &_data[index+count];
            return retval;
         }

         DEVICEONLY
         iterator device_insert (iterator it, const T& val)noexcept{
            
            //If empty or inserting at the end no relocating is needed
            if (it==end()){
               device_push_back(val);
               return end()--;
            }

            int64_t index=it.data()-begin().data();
            if (index<0 || index>size()){
               assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no space available.");
            }

            if (size() == capacity()) {
               assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no space available.");
            }

            //Increase size;
            for(int64_t  i = size() - 1; i >= index; i--){
               _data[i+1] = _data[i];
            }
            _data[index] = val;
            device_resize(size()+1);
            return iterator(_data+index);
         }

         DEVICEONLY
         iterator device_insert(iterator it,const size_t elements, const T& val){


            int64_t index=it.data()-begin().data();
            size_t oldsize=size();
            size_t newSize=oldsize+elements;
            if (index<0 || index>size()){
               assert(0 && "Splitvector has a catastrophic failure trying to insert on device because the vector has no space available.");
            }
            
            //Do we do need to increase our capacity?
            if (newSize>size()){
               device_resize(newSize);
            }

            it = begin().data() + index;
            iterator last= it.data()+oldsize;
            iterator target=last.data()+elements;
            for (iterator candidate=last; candidate!=it; --candidate){
               *target=*candidate;
               --target;
            }

            last= it.data()+elements;
            //std::fill(it,last,val);
            target=last;
            for (iterator candidate=it; candidate!=last; ++candidate){
               *target=val;
               ++target;
            }

            target = &_data[index];
            return target;
         }

         #endif

         HOSTDEVICE 
         iterator erase(iterator it)noexcept{
            const int64_t index = it.data() - begin().data();
            _data[index].~T();

            for (auto i = index; i < size() - 1; i++) {
               new (&_data[i]) T(_data[i+1]); 
               _data[i+1].~T();
            }
            *_size-=1;
            iterator retval = &_data[index];
            return retval;   
         }
            

         HOSTDEVICE
         iterator erase(iterator p0, iterator  p1)noexcept {
            const int64_t start = p0.data() - begin().data();
            const int64_t end   =  p1.data() - begin().data();
            const int64_t offset= end- start;

            for (int64_t i = 0; i < end - start; i++) {
               _data[i].~T();
            }
            for (auto i =start; i < size()-offset; ++i){
               new (&_data[i]) T(_data[i+offset]); 
               _data[i+offset].~T();
            }
            *_size -= end - start;
            iterator it = &_data[start];
            return it;
         }


         HOSTONLY 
         Allocator get_allocator() const noexcept{
            return _allocator;
         }

         template< class... Args >
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

         template< class... Args >
         void emplace_back(Args&&... args) {
            emplace(end(), std::forward<Args>(args)...);
         }
   };//SplitVector

	/*Equal operator*/
	template <typename  T,class Allocator,class Meta_Allocator>
	static inline HOSTDEVICE  bool operator == (const  SplitVector<T,Allocator,Meta_Allocator> &lhs, const  SplitVector<T,Allocator,Meta_Allocator> &rhs) noexcept{
		if (lhs.size()!= rhs.size()){
			return false;
		}
		for (size_t i=0; i<lhs.size(); i++){
			if ( !(lhs[i]==rhs[i]) ){
				return false;
			}
		}
		//if we end up here the vectors are equal
		return true;
	}

	/*Not-Equal operator*/
	template <typename  T,class Allocator,class Meta_Allocator>
	static inline HOSTDEVICE  bool operator != (const  SplitVector<T,Allocator,Meta_Allocator> &lhs, const  SplitVector<T,Allocator,Meta_Allocator> &rhs) noexcept{
		return !(rhs==lhs);
	}
}//namespace split

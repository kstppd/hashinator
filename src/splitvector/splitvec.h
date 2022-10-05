#pragma once
/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
#include <iostream>
#include <cuda_runtime_api.h>
#include "split_allocators.h"
#include <cuda.h>
#include <cassert>
#include <cstring>
#include <stdlib.h>
#include <algorithm>
#include <memory>


namespace split{


   template <typename T> void swap(T& t1, T& t2) {
       T tmp = std::move(t1);
       t1 = std::move(t2);
       t2 = std::move(tmp);
   }

   template<typename T,class Allocator=split::split_host_allocator<T>,class Meta_Allocator=split::split_host_allocator<size_t>>
   class SplitVector{
      
      private:
         T* _data=nullptr;                   //actual pointer to our data      
         size_t* _size;                      // number of elements in vector.
         size_t* _capacity;                  // number of allocated elements
         size_t  _alloc_multiplier = 2;      //host variable; multiplier for  when reserving more space
         Allocator  _allocator;              // Allocator used to allocate and deallocate memory;
         Meta_Allocator  _meta_allocator;    // Allocator used to allocate and deallocate memory for metadata;
 
         void _check_ptr(void* ptr){
            if (ptr==nullptr){
               throw std::bad_alloc();
            }
         }

         /*Internal range check for use in .at()*/
         __host__ __device__ void _rangeCheck(size_t index){
            assert(index<size() &&  "out of range");
         }

         /*Allocation/Deallocation only on host*/
         void _allocate(size_t size){
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

         void _deallocate(){
               if (_data!=nullptr){
                  _deallocate_and_destroy(capacity(),_data);
                  _data=nullptr;
               }
               _deallocate_and_destroy(_capacity);
               _deallocate_and_destroy(_size);
         }

         T* _allocate_and_construct(size_t n, const T &val){
            T* _ptr=_allocator.allocate(n);
            for (size_t i=0; i < n; i++){
               _allocator.construct(&_ptr[i],val);
            }
            return _ptr;
         }
         size_t* _allocate_and_construct(const size_t &val){
            size_t* _ptr=_meta_allocator.allocate(1);
            _meta_allocator.construct(_ptr,val);
            return _ptr;
         }

         void _deallocate_and_destroy(size_t n,T* _ptr){
            for (size_t i=0; i < n; i++){
               _allocator.destroy(&_ptr[i]);
            }
            _allocator.deallocate(_ptr,n);
         }

         void _deallocate_and_destroy(size_t* ptr){
            _meta_allocator.deallocate(ptr,1);
         }

      public:
         /* Available Constructors :
          *    -- SplitVector()                       --> Default constructor. Almost a no OP but _size and _capacity have  to be allocated for device usage. 
          *    -- SplitVector(size_t)                 --> Instantiates a splitvector with a specific size. (capacity == size)
          *    -- SplitVector(size_t,T)               --> Instantiates a splitvector with a specific size and sets all elements to T.(capacity == size)
          *    -- SplitVector(SplitVector&)           --> Copy constructor. 
          *    -- SplitVector(std::initializer_list&) --> Creates a SplitVector and copies over the elemets of the init. list. 
          *    -- SplitVector(std::vector&)           --> Creates a SplitVector and copies over the elemets of the std vector
          * */

         /*Constructors*/
         __host__ explicit   SplitVector(){
            this->_allocate(0); //seems counter-intuitive based on stl but it is not!
         }

         __host__ explicit   SplitVector(size_t size){
               this->_allocate(size);
         }

         __host__ explicit  SplitVector(size_t size, const T &val){
               this->_allocate(size);
               for (size_t i=0; i<size; i++){
                  _data[i]=val;
               }
            }

         __host__ SplitVector(const SplitVector<T,Allocator,Meta_Allocator> &other){
               const size_t size_to_allocate = other.size();
               this->_allocate(size_to_allocate);
               for (size_t i=0; i<size_to_allocate; i++){
                  _data[i]=other._data[i];
               }
            }

         __host__ SplitVector(std::initializer_list<T> init_list){
               this->_allocate(init_list.size());
               for (size_t i =0 ; i< size();i++){
                  _data[i]=init_list.begin()[i];
               }
            }
    
         __host__ explicit  SplitVector(const std::vector<T> &other ){
               this->_allocate(other.size());
               for (size_t i=0; i<size(); i++){
                  _data[i]=other[i];
               }
            }
         
         //Destructor
         __host__ ~SplitVector(){
            _deallocate();
         }

         
         /*Custom Assignment operator*/
         __host__  SplitVector<T,Allocator,Meta_Allocator>& operator=(const SplitVector<T,Allocator,Meta_Allocator>& other){
            //Match other's size prior to copying
            resize(other.size());
            for (size_t i=0; i< size(); i++){
               _data[i]=other._data[i];
            }
            return *this;
         }


         //Method that return a pointer which can be passed to GPU kernels
         //Has to be cudaFree'd after use otherwise memleak (small one but still)!
         __host__
         SplitVector<T,Allocator,Meta_Allocator>* upload(cudaStream_t stream = 0 ){
            SplitVector* d_vec;
            optimizeGPU(stream);
            cudaMalloc((void **)&d_vec, sizeof(SplitVector));
            cudaMemcpyAsync(d_vec, this, sizeof(SplitVector),cudaMemcpyHostToDevice,stream);
            return d_vec;
         }

         /*Manually prefetch data on Device*/
         __host__ void optimizeGPU(cudaStream_t stream = 0){
            int device;
            cudaGetDevice(&device);
            CheckErrors("Prefetch GPU-Device-ID");
            cudaMemPrefetchAsync(_data ,capacity()*sizeof(T),device,stream);
            CheckErrors("Prefetch GPU");
         }

         /*Manually prefetch data on Host*/
         __host__ void optimizeCPU(cudaStream_t stream = 0){
            cudaMemPrefetchAsync(_data ,capacity()*sizeof(T),cudaCpuDeviceId,stream);
            CheckErrors("Prefetch CPU");
         }

         /*Custom swap mehtod. Pointers invalidated afters swap. 
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
         __host__ __device__ const size_t& size() const{
            return *_size;
         }

         /*Bracket accessor - no bounds check*/
         __host__ __device__ T& operator [](size_t index){
               return _data[index];
         } 
                  
         /*Const Bracket accessor - no bounds check*/
         __host__ __device__ const T& operator [](size_t index)const{
               return _data[index];
         } 

         /*at accesor with bounds check*/
         __host__ __device__ T& at(size_t index){
            _rangeCheck(index);
            return _data[index];
         }
         
         /*const at accesor with bounds check*/
         __host__ __device__ const T& at(size_t index)const{
            _rangeCheck(index);
            return _data[index];
         }

         /*Return a raw pointer to our data similar to stl vector*/
         __host__ __device__ T* data(){
            return &(_data[0]);
         }


         /* Size Modifiers
            Reserve method:
            Supports only host reserving.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalidated after a call.
         */
         __host__
         void reserve(size_t requested_space){
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
               return ;
            }
            
            requested_space*=_alloc_multiplier;
            // Allocate new Space
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
            return;
         }

          /* 
            Resize method:
            Supports only host resizing.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalid from now on.
         */
         __host__
         void resize(size_t newSize){
            //Let's reserve some space and change our size
            reserve(newSize);
            *_size  =newSize; 
         }

         __host__
         void grow(){
            reserve(capacity()+1);
         }
 
         __host__
         void shrink_to_fit(){
            size_t curr_cap =*_capacity;
            size_t curr_size=*_size;

            if (curr_cap == curr_size){
               return;
            }

            // Allocate new Space
            T* _new_data;
            _new_data=_allocate_and_construct(curr_size,T());
            if (_new_data==nullptr){
               _deallocate_and_destroy(curr_size,_new_data);
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
            *_capacity=size();
            return;
         }
         
         /*Removes the last element of the vector*/
         __host__ __device__
         void pop_back(){
            if (size()>0){
               remove_from_back(1);
            }
            return;
         }

         //Removes n elements from the back of the vector\
         //and properly handles object destruction
         __host__ __device__
        void remove_from_back(size_t n){
          const size_t end = size() - n;
          for (auto i = size(); i > end;) {
            (_data + --i)->~T();
          }
          *_size = end;
        }

         __host__
         void clear(){
             for (size_t i = 0; i < size();i++) {
               _data[i].~T();
             }
            *_size=0;
            return;
         }

         __host__ __device__
         size_t capacity() const {
            return *_capacity;
         }

         __host__ __device__
         T& back(){ return _data[*_size-1]; }

         __host__ __device__
         const T& back() const{return _data[*_size-1];}
         
         __host__ __device__
         T& front(){return _data[0];}
         
         __host__ __device__
         const T& front() const{ return _data[0]; }

         __host__ __device__ 
         bool empty() const{
            return  size()==0;
         }

         #ifndef __CUDA_ARCH__
         /* 
            PushBack  method:
            Supports only host  pushbacks.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalid from now on.
            Not thread safe
         */      
         __host__   
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
         
         #else         

         __device__ 
         void push_back(const T& val){
            //We need at least capacity=size+1 otherwise this 
            //pushback cannot be done
            size_t old= atomicAdd((unsigned int*)_size, 1);
            if (old>=capacity()){
               assert(0 && "Splitvector has a catastrophic failure trying to pushback on device because the vector has no space available.");
            }
            atomicCAS(&(_data[old]), _data[old],val);
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
            iterator(pointer data) : _data(data) {}

            pointer data() { return _data; }
            pointer operator->() { return _data; }
            reference operator*() { return *_data; }

            bool operator==(const iterator& other)const{
              return _data == other._data;
            }
            bool operator!=(const iterator& other)const {
              return _data != other._data;
            }
            iterator& operator++(){
              _data += 1;
              return *this;
            }
            iterator operator++(int){
              return iterator(_data + 1);
            }
            iterator operator--(int){
              return iterator(_data - 1);
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
            using reference = T&;

            const_iterator(pointer data) : _data(data) {}

            pointer data()const { return _data; }
            pointer operator->()const  { return _data; }
            reference operator*()const  { return *_data; }

            bool operator==(const const_iterator& other)const{
              return _data == other._data;
            }
            bool operator!=(const const_iterator& other)const {
              return _data != other._data;
            }
            const_iterator& operator++(){
              _data += 1;
              return *this;
            }
            const_iterator operator++(int){
              return const_iterator(_data + 1);
            }
            const_iterator operator--(int){
              return const_iterator(_data - 1);
            }
         };
         
         iterator begin(){
            return iterator(_data);
         }

         const_iterator begin()const{
            return const_iterator(_data);
         }

         iterator end(){
            return iterator(_data+size());
         }
         const_iterator end() const {
            return const_iterator(_data+size());
         }

         __host__
         iterator insert (iterator& it, const T& val){
            
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

         __host__
         iterator insert(iterator& it,const size_t elements, const T& val){

            int64_t index=it.data()-begin().data();
            if (index<0 || index>size()){
               throw std::out_of_range("Insert");
            }
            
            //Do we do need to increase our capacity?
            if (size()+elements>capacity()){
               resize(capacity()+elements);
            }

            iterator retval = &_data[index];
            std::move(retval, end(), retval.data() + elements);
            std::fill_n(retval, elements, val);
            return retval;
         }


         __host__
         iterator insert(iterator& it, iterator p0, iterator p1) {
         
            const int64_t count = p1.data() - p0.data();
            const int64_t index = it.data() - begin().data();
      
            if (index<0 || index>size()){
               throw std::out_of_range("Insert");
            }

            if (size() + count > capacity()) {
               resize(capacity() + count);
            }

            iterator retval = &_data[index];
            std::move(retval, end(), retval.data() + count);
            std::copy(p0, p1, retval);
            return retval;
         }
         

         iterator erase(iterator& it){

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
            

         __host__ 
         iterator erase(iterator& p0, iterator& p1) {
            const int64_t start = p0.data() - begin().data();
            const int64_t end   =  p1.data() - begin().data();

            for (int64_t i = 0; i < end - start; i++) {
               _data[start+i].~T();
               new (&_data[start+i]) T(_data[end+1]); 
            }

            *_size -= end - start;
            iterator it = &_data[start];
            return it;
         }

         Allocator get_allocator() const {
            return _allocator;
         }


   };
    
	/*Equal operator*/
	template <typename  T,class Allocator,class Meta_Allocator>
	static inline __host__  bool operator == (const  SplitVector<T,Allocator,Meta_Allocator> &lhs, const  SplitVector<T,Allocator,Meta_Allocator> &rhs){
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
	static inline __host__  bool operator != (const  SplitVector<T,Allocator,Meta_Allocator> &lhs, const  SplitVector<T,Allocator,Meta_Allocator> &rhs){
		return !(rhs==lhs);
	}
}

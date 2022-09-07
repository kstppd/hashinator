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
#include <cuda.h>
#include <cassert>
#include <cstring>

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
#pragma message ("TODO-->Make this a no NOOP" )
   #define CheckErrors(msg) \
      do { }  while (0)
#endif

namespace split{
   


   template<typename T>
   class SplitVector{
      
      private:
         T*      _data; //actual pointer to our data      
         size_t* _size; // number of elements in vector.
         size_t* _capacity; // number of allocated elements
         size_t  _alloc_multiplier = 1; //host variable; multiplier for  when reserving more space
 

#ifdef CUDAVEC
         /*Allocation/Deallocation with unified memory*/
         void _allocate(size_t size){
               cudaMallocManaged((void**)&_data, size * sizeof(T));
               cudaMallocManaged((void**)&_size,sizeof(size_t));
               cudaMallocManaged((void**)&_capacity,sizeof(size_t));
               *this->_size= size;
               *this->_capacity= size;
               CheckErrors("Managed Allocation");
               //Here we also need to construct the object
               for (size_t i=0; i < size; i++){
                  new (&_data[i]) T(); 
               }
               CheckErrors("Managed Allocation");
               cudaDeviceSynchronize();
         }

         void _deallocate(){
               if (_data!=nullptr){
                  for (size_t i=0; i<size();i++){
                     _data[i].~T();
                  }
                  cudaFree(_data);
                  _data=nullptr;
                  CheckErrors("Managed Deallocation");
               }
               cudaFree(_size);
               CheckErrors("Managed Deallocation");
               cudaFree(_capacity);
               CheckErrors("Managed Deallocation");
               cudaDeviceSynchronize();
         }

#else
         /*Allocation/Deallocation only on host*/
         void _allocate(size_t size){
            _data=new T[size];
            _size=new size_t(size);
            _capacity=new size_t(size);
            if (_data == nullptr){
               delete [] _data;
               delete _size;
               throw std::bad_alloc();
            }
         }

         void _deallocate(){
               delete _size;
               delete _capacity;
               if (_data!=nullptr){
                  delete [] _data;
                  _data=nullptr;
               }
         }
#endif

      public:

         /*Constructors*/
         __host__ explicit   SplitVector():_data(nullptr){
               #ifdef CUDAVEC
               cudaMallocManaged((void**)&_size,sizeof(size_t));
               cudaMallocManaged((void**)&_capacity,sizeof(size_t));
               *this->_size= 0;
               *this->_capacity=0;
               #else
               _size=new size_t(0);
               _capacity=new size_t(0);
               #endif
         }

         __host__ explicit   SplitVector(size_t size)
               :_data(nullptr){
               this->_allocate(size);
         }

         __host__ explicit  SplitVector(size_t size, const T &val)
               :_data(nullptr){
               this->_allocate(size);
               for (size_t i=0; i<size; i++){
                  _data[i]=val;
               }
            }

         __host__ SplitVector(const SplitVector &other){
            std::cout<<"Copy constructor init"<<std::endl;
               const size_t size_to_allocate = other.size();
               this->_allocate(size_to_allocate);
               for (size_t i=0; i<size_to_allocate; i++){
                  _data[i]=other._data[i];
               }
            }

         __host__ SplitVector(std::initializer_list<T> init_list)
               :_data(nullptr){
               this->_allocate(init_list.size());
               for (size_t i =0 ; i< size();i++){
                  _data[i]=init_list.begin()[i];
               }
            }
         
         //Destructor
         __host__ ~SplitVector(){
            _deallocate();
         }

         
         /*Custom Assignment operator*/
         __host__  SplitVector& operator=(const SplitVector& other){
            if (size() == other.size()){
               for (size_t i=0; i< size(); i++){
                  _data[i]=other._data[i];
               }
            }else{
               // we need to allocate a new block unfortunately
               _deallocate();
               _allocate(other.size());
               for (size_t i=0; i< size(); i++){
                  _data[i]=other._data[i];
               }
            }
            return *this;
         }

         /*Custom swap mehtod. Pointers after swap 
         * are pointing to the same container as 
         * before. 
         */
         void swap(SplitVector<T>& other) noexcept{

            if (*this==other){ //no need to do any work
               return;
            }
            SplitVector<T> temp(this->size());
            temp=*this;
            *this=other;
            other=temp;
            return;
         }

         /*Returns number of elements in this container*/
         __host__ __device__ const size_t& size() const{
            return *_size;
         }

         /*Internal range check for use in .at()*/
         __host__ __device__ void rangeCheck(size_t index){
            assert(index<size() &&  "out of range");
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
            rangeCheck(index);
            return _data[index];
         }
         
         /*const at accesor with bounds check*/
         __host__ __device__ const T& at(size_t index)const{
            rangeCheck(index);
            return _data[index];
         }

         /*Return a raw pointer to our data similar to stl vector*/
         __host__ __device__ T* data(){
            return &(_data[0]);
         }

         /*Manually prefetch data on Device*/
         __host__ void optimizeGPU(cudaStream_t stream = 0){
            int device;
            cudaGetDevice(&device);
            CheckErrors("Prefetch GPU-Device-ID");
            cudaMemPrefetchAsync(_data ,size()*sizeof(T),device,stream);
            CheckErrors("Prefetch GPU");
         }

         /*Manually prefetch data on Host*/
         __host__ void optimizeCPU(cudaStream_t stream = 0){
            cudaMemPrefetchAsync(_data ,size()*sizeof(T),cudaCpuDeviceId,stream);
            CheckErrors("Prefetch CPU");
         }

         //Size modifiers
         /* 
            Reserve method:
            Supports only host reserving.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalidated after a call.
         */
         __host__
         void reserve(size_t requested_space){
            size_t current_space=*_capacity;
            
            //Nope.
            if (requested_space <= current_space){
               return ;
            }
            
            requested_space*=_alloc_multiplier;
            _alloc_multiplier*=2;
            // Allocate new Space
            T* _new_data;
            #ifdef CUDAVEC
               cudaMallocManaged((void**)&_new_data, requested_space * sizeof(T));
               //Here we also need to construct the objects manually
               for (size_t i=0; i < requested_space; i++){
                  new (&_new_data[i]) T(); 
               }
               CheckErrors("Reserve:: Managed Allocation");
            #else
               _new_data=new T[requested_space];
               if (_new_data==nullptr){
                  delete [] _new_data;
                  this->_deallocate();
                  throw std::bad_alloc();
               }
            #endif
            
            //Copy over
            for (size_t i=0; i<size();i++){
               _new_data[i] = _data[i];
            }

            //Deallocate old space
            #ifdef CUDAVEC
               cudaFree(_data);
            #else
               delete [] _data;
            #endif

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

       /* 
            PushBack  method:
            Supports only host  pushbacks.
            Will never reduce the vector's size.
            Memory location will change so any old pointers/iterators
            will be invalid from now on.
            Not thread safe
         */      
         __host__   
         void push_back(T val){
            // If we have no allocated memory because the default ctor was used then 
            // allocate one element, set it and return 
            if (_data==nullptr){
               _allocate(1);
               _data[size()-1] = val;
               return;
            }

            resize(size()+1);
            _data[size()-1] = val;
            return;
         }
         
         /*
            Trim method:
            Will never reduce the actual number of elements
            in the vector but only deallocate the excess elements 
            and try to get capacity to match size.
         */
         __host__
         void trim(){
            
            //Cannot trim unallocated space
            if (_data==nullptr){
               return;
            }

            //No chance to trim
            if (size()==*_capacity){
               return ;
            }

            // Allocate new Space
            T* _new_data;
            #ifdef CUDAVEC
               cudaMallocManaged((void**)&_new_data, size() * sizeof(T));
               for (size_t i=0; i < size(); i++){
                  new (&_new_data[i]) T(); 
               }
               CheckErrors("Reserve:: Managed Allocation");
            #else
               _new_data=new T[size()];
               if (_new_data==nullptr){
                  delete [] _new_data;
                  this->_deallocate();
                  throw std::bad_alloc();
               }
            #endif
            
             //Copy over
            for (size_t i=0; i<size();i++){
               _new_data[i] = _data[i];
            }

            //Deallocate old space
            #ifdef CUDAVEC
               cudaFree(_data);
            #else
               delete [] _data;
            #endif

            //Swap pointers & update capacity
            //Size remains the same ofc
            _data=_new_data;
            *_capacity=size();
            //Reset multiplier
            _alloc_multiplier=1;
            return;
         }

         /*Removes the last element of the vector*/
         __host__ __device__
         void pop_back(){
            if (size()>0){
               *_size=*_size-1;
            }
            return;
         }

         __host__ __device__
         size_t capacity(){
            return *_capacity;
         }


            //~Size modifiers

         /*Host side operator += */
         __host__  SplitVector<T>& operator+=(const SplitVector<T>& rhs) {
            assert(this->size()==rhs.size());
            for (size_t i=0; i< this->size(); i++){
               this->_data[i] = this->_data[i]+rhs[i];
            }
            return *this;
         }


         /*Host side operator /= */
         __host__ SplitVector<T>& operator/=(const SplitVector<T>& rhs) {
            assert(this->size()==rhs.size());
            for (size_t i=0; i< this->size(); i++){
               this->_data[i] = this->_data[i]/rhs[i];
            }
            return *this;
         }

         /*Host side operator -= */
         __host__ SplitVector<T>& operator-=(const SplitVector<T>& rhs) {
            assert(this->size()==rhs.size());
            for (size_t i=0; i< this->size(); i++){
               this->_data[i] = this->_data[i]-rhs[i];
            }
            return *this;
         }

         /*Host side operator *= */
         __host__ SplitVector<T>& operator*=(const SplitVector<T>& rhs) {
            assert(this->size()==rhs.size());
            for (size_t i=0; i< this->size(); i++){
               this->_data[i] = this->_data[i]*rhs[i];
            }
            return *this;
         }


         //Iterators
         class iterator{
             
            private:
            T* _data;
            
            public:
            
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = size_t;
            using pointer = T*;
            using reference = T&;

            //iterator(){}
            iterator(pointer data) : _data(data) {}

            pointer data() { return _data; }
            reference operator*() { return *_data; }
            bool operator!=(const iterator& other){
              return _data != other._data;
            }
            iterator& operator++(){
              _data += 1;
              return *this;
            }
            iterator operator++(int){
              return iterator(_data + 1);
            }
         };


         class const_iterator{
            private: 
            T* _data;
            
            public:
            
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = size_t;
            using pointer = T*;
            using reference = T&;

            //const_iterator(){}
            explicit const_iterator(pointer data) : _data(data) {}

            const pointer data() const { return _data; }
            const reference operator*() const  { return *_data; }
            bool operator!=(const const_iterator& other){
              return _data != other._data;
            }
            const_iterator& operator++(){
              _data += 1;
              return *this;
            }
            const_iterator operator++(int){
              return const_iterator(_data + 1);
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

         //***************************temp****************************  
         __host__  void print(){
               std::cout<<&(_data[0])<<" -->"<<_data[0]<<" - "<<_data[size()-1 ]<<" Capacity= "<<*_capacity<<" Size= "<<size()<<std::endl;
         } 

         __host__  void print_full(){
            for (size_t i =0; i< size(); i++){
               std::cout<<at(i)<<std::endl;
            } 
         } 


   };
    
   template <typename  T>
   static inline __host__  SplitVector<T> operator+(const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
      SplitVector<T> child(lhs.size());
      child=lhs;
      return child+=rhs;
   }

   template <typename  T>
   static inline __host__  SplitVector<T> operator-(const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
      SplitVector<T> child(lhs.size());
      child=lhs;
      return child-=rhs;
   }

   template <typename  T>
   static inline __host__  SplitVector<T> operator*(const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
      SplitVector<T> child(lhs.size());
      child=lhs;
      return child*=rhs;
   }
 
   template <typename  T>
   static inline __host__  SplitVector<T> operator/(const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
      SplitVector<T> child(lhs.size());
      child=lhs;
      return child/=rhs;
   }
	
	/*Equal operator*/
	template <typename  T>
	static inline __host__  bool operator == (const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
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
	template <typename  T>
	static inline __host__  bool operator != (const  SplitVector<T> &lhs, const  SplitVector<T> &rhs){
		return !(rhs==lhs);
	}
}

#pragma once
#include <iostream>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


class DoublyStackedArena{
using val_type = char;
private:
   //Memebrs;
   val_type*  _data;
   val_type** _start;
   val_type** _current;
   size_t *_size;
   size_t *_available;
   
   __host__
   void reset(){
      _start[0] = _data;
      _start[1] = _data + *_size;
      _current[0] = _start[0];
      _current[1] = _start[1];
      *_available=*_size;
   }

public:
   __host__
   DoublyStackedArena(size_t size){
      cudaMallocManaged((val_type**)(&(_start)), 2);
      cudaMallocManaged((val_type**)(&(_current)), 2);
      cudaMallocManaged((size_t**)(&(_size)),1);
      cudaMallocManaged((size_t**)(&(_available)),1);
      *_size=size;
      *_available=size;
      cudaMallocManaged((val_type**)(&(_data)), *_size);
      this->reset();
   }


   __host__
   void cleanup(){
      cudaFree (_data);
      cudaFree (_start);
      cudaFree (_current);
      cudaFree (_size);
      cudaFree (_available);

   }

   __host__ __device__      
   size_t size()const {return *_size;}

   __host__ __device__      
   size_t availableSpace()const {return *_available;}

   __host__ __device__      
   bool canAllocate()const {
      if ( _current[0]!=_start[0] && 
           _current[1]!=_start[1])
      {
         return false;
      }
      return true;
   }
   __host__ __device__
   uintptr_t pad(uintptr_t size, std::size_t alignment){
      const size_t mask = alignment - 1;
      return (size + mask) & ~mask;
   }

   __host__ __device__      
   void* allocate(size_t size,size_t align=8){
      
      size=pad(size,align);
      //Not enough space
      if ( _current[0] + size >_current[1]){
         return nullptr;
      }
      if (_current[0]==_start[0]){
         _current[0]+=size;
         *_available-=size;
#ifdef DEBUG
         printf("Left::Size /Free  = %zu / %zu \n",this->size(),this->availableSpace());
#endif
         return (void*)_start[0];
      }
      if (_current[1]==_start[1]){
         _start[1]-=size;
         *_available-=size;
#ifdef DEBUG
         printf("Right::Size /Free  = %zu / %zu \n",*_size,*_available);
#endif
         return (void*)_start[1];
      }
      //Probably we already have 2 allocations in use so return a nullptr
      return nullptr;
   }
   
   __host__ __device__
   void deallocate(void * ptr){
      if (ptr==_start[0]){
         *_available +=abs(_current[0]-_start[0]);
         _current[0]=_start[0];
      }
      if (ptr==_start[1]){
         *_available += abs(_current[1]-_start[1]);
         _current[1]=_start[1];
      }
   }

   void dump(uint16_t stride = 4){
      for (size_t i=0;i<this->size();i+=stride){
         printf("%i ",(int)*(_data+i));
      }
   }


};

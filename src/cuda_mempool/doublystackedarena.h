#pragma once
#include <iostream>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


class DoublyStackedArena{
   using uchar = unsigned char;
   private:
      //Memebrs;
      uchar*  _data;
      uchar** _start;
      uchar** _current;
      size_t *_size=nullptr;
      size_t *_available=nullptr;
      
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
         cudaMallocManaged((uchar**)(&(_start)), 2);
         cudaMallocManaged((uchar**)(&(_current)), 2);
         cudaMallocManaged((size_t**)(&(_size)),1);
         cudaMallocManaged((size_t**)(&(_available)),1);
         *_size=size;
         *_available=*_size;
         cudaMallocManaged((uchar**)(&(_data)), *_size);
         this->reset();
      }

      __host__
      ~DoublyStackedArena(){
         cudaFree (_data);
         cudaFree (_start);
         cudaFree (_current);
         cudaFree (_size);
         cudaFree (_available);
      }

      size_t Size(){return *_size;}
      size_t AvailableSpace(){return *_available;}
      __host__ __device__
      uintptr_t pad(uintptr_t size, std::size_t alignment){
         const size_t mask = alignment - 1;
         return (size + mask) & ~mask;
      }

      __host__ __device__      
      void* allocate(size_t size,size_t align=8){
         //Not enough space
         if ( _current[0] + size >_current[1]){
            return nullptr;
         }
         size=pad(size,align);
         if (_current[0]==_start[0]){
            _current[0]+=size;
            *_available-=size;
            return (void*)_start[0];
         }
         if (_current[1]==_start[1]){
            _current[1]-=size;
            *_available-=size;
            return (void*)_start[1];
         }
         //Probably we already have 2 allocations in use so return a nullptr
         return nullptr;
      }
      
      __host__ __device__
      void deallocate(void * ptr){
         if (ptr==_start[0]){
            _current[0]=_start[0];
         }
         if (ptr==_start[1]){
            _current[1]=_start[1];
         }
      }
};
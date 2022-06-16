#pragma once
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "list.h"
#include "doublystackedarena.h"




class UnifiedMemPool{

private:
   typedef  DoublyStackedArena DSA;
   typedef FixedList<DSA>::Node Node;
   FixedList<DSA> nodeList;



public:

   UnifiedMemPool(){}
   void* allocate(size_t size){
      Node* block=nodeList.head;
      while(block!=nullptr){
#ifdef DEBUG
         std::cout<<"Looking at Block= "<<block<<std::endl;
#endif
         if (!block->alive){
#ifdef DEBUG
            std::cout<<"EmptySpot"<<std::endl;
#endif
            block->data = DSA(3*size);
            block->alive=true;
            void* memblock = block->data.allocate(size);
            return memblock;
         }   

         if (block->data.availableSpace()>= size && block->data.canAllocate()){
#ifdef DEBUG
            std::cout<<"Half-EmptySpot"<<std::endl;
#endif
            void* memblock = block->data.allocate(size);
            return memblock;
         }

         block=block->next;
         
      }
      return nullptr;

   }


   void free(void* ptr){
      Node* block=nodeList.head;
      while(block!=nullptr){
         if (block->alive){
            block->data.deallocate(ptr);
         }
         block=block->next;
      }
      return;
   }



   __host__ 
   void stats(uint16_t stride= 4){
      std::cout<<"***************STATS*******************"<<std::endl;
      Node* block=nodeList.head;
      while(block!=nullptr){

         if (!block->alive){
            std::cout<<"Dead Block= "<<block<<std::endl;
         }else{
            std::cout<<"Alive Block= "<<block<<" -- Size info "<<block->data.size()<<" / "<<block->data.availableSpace()<<std::endl;
            std::cout<<"--------------MEMORY DUMP--------------\n"<<std::endl;
            block->data.dump(stride);
            std::cout<<"\n\n---------------------------------------"<<std::endl;

         }
         block=block->next;
      }
   }
};

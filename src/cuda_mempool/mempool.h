#pragma once
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "list.h"
#include "doublystackedarena.h"



typedef  DoublyStackedArena DSA;

class UnifiedMemPool{


private:

   FixedList<DSA> nodeList;
   typedef FixedList<DSA>::Node Node;



public:

   UnifiedMemPool(){}

   void* allocate(size_t size){

      Node* block=nodeList.head;
      while(block!=nullptr){
         std::cout<<"Looking at Block= "<<block<<std::endl;
         if (!block->alive){
            std::cout<<"EmptySpot"<<std::endl;
            block->data=DSA(3*size);
            block->alive=true;
            void* memblock = block->data.allocate(size);
            if (memblock!=nullptr){
               return memblock;
            }
         }   
         
         void* memblock = block->data.allocate(size);
         if (memblock!=nullptr){
            return memblock;
         }

         block=block->next;
      }

   }


   __host__ 
   void stats(){
      Node* block=nodeList.head;
      while(block!=nullptr){

         if (!block->alive){
            std::cout<<"Dead Block= "<<block<<std::endl;
         }else{
            std::cout<<"Alive Block= "<<block<<std::endl;
            std::cout<<"Size= "<<block->data.Size()<<std::endl;
            std::cout<<"Available= "<<block->data.AvailableSpace()<<std::endl;

         }
         block=block->next;
      }
   }







};

#pragma once
#include <iostream>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define N 4

template<typename T>
class FixedList{
public:

   struct Node{
      T data;
      Node* next;
      bool alive=false;
   };



   FixedList(){
      cudaMallocManaged((void**)(&(current)), sizeof(Node));
      head=nullptr;
      current=nullptr;


      for (int i =0; i<N; i++){
         Node* newNode;
         cudaMallocManaged((void**)(&(newNode)), sizeof(Node));
         newNode->next=nullptr;
         if (head==nullptr){
            head=newNode;
         }else{
            newNode->next=head;
            head=newNode;
         }
      }
      // print();
   }
      
   ~FixedList(){
      current=head;
      while (current!=nullptr){
         Node* toKill=current;
         current=current->next;
         // std::cout<<"Killing node-> "<<toKill<<std::endl;
         cudaFree(toKill);
      }
      cudaFree(current);
   }

   void print(){
      current=head;
      while (current!=nullptr){
         std::cout<<"New node at -> "<<current<<std::endl;
         current=current->next;
      }
   }

   Node* head;
   Node* current;

};
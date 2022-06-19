#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>

#define IS_TRUE(x)\
   { if (!(x)) {\
   std::cout << __FUNCTION__ << "\033[1;31m failed on line \033[0m " << __LINE__<<std::endl;\
   exit(1);\
   }else{\
   std::cout << "\033[1;32m ****OK**** \033[0m\n"<<std::endl; \
   }\
} 


typedef split::SplitVector<int> vec ;
typedef split::SplitVector<double> dvec ;

__global__
void gpu_add(vec a ,vec b , vec c ){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < a.size(); i += stride)
    c.at(i)= a.at(i)+ b.at(i);
}

__global__
void gpu_add_double(dvec a ,dvec b , dvec c ){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < a.size(); i += stride)
    c.at(i)= a.at(i)+ b.at(i);
}



bool swap_same_size(){
   const int N=1e8;
   vec a(N);
   vec b(N);
   std::fill(a.begin(),a.end(),1);
   std::fill(b.begin(),b.end(),2);
   std::cout<<"a size="<<a.size()<<std::endl;
   std::cout<<"b size="<<b.size()<<std::endl;   
   int* ad_a=a.data();
   int* ad_b=b.data();
   a.print();
   b.print();
   std::cout<<"A resides in --> "<<ad_a<<std::endl;
   std::cout<<"B resides in --> "<<ad_b<<std::endl;
   a.swap(b);
   a.print();
   b.print();
   int* ad_a_2=a.data();
   int* ad_b_2=b.data();
   std::cout<<"A resides in --> "<<ad_a_2<<std::endl;
   std::cout<<"B resides in --> "<<ad_b_2<<std::endl;
   

   for ( auto elem:a ){
      if (elem!=2){
         return false;
      }
   } 

   for ( auto elem:b ){
      if (elem!=1){
         return false;
      }
   } 
   return true;
}



bool swap_different_size(){
   const int N=1e8;
   const int N2=2e8;
   vec a(N);
   vec b(N2);
   std::fill(a.begin(),a.end(),1);
   std::fill(b.begin(),b.end(),2);
   try { 
      a.print();
      b.print();
      a.swap(b);
      a.print();
      b.print();
      b.swap(a);
      a.print();
      b.print();
   } catch (...) { 
      std::cout << "There was a catastrophic exception of some kind at "<<__FILE__<<" : "<<__LINE__<<std::endl; 
      return false;
   }

   for ( auto elem:a ){
      if (elem!=1){
         return false;
      }
   } 

   for ( auto elem:b ){
      if (elem!=2){
         return false;
      }
   } 

   if (a.size()!=N || b.size()!=N2){
      return false;
   }
   return true;
}

bool gpu_add(){
   const int N=1e8;
   //declare 3 vectors 
   vec a(N);
   vec b(N);
   vec c(N);
   std::fill(a.begin(),a.end(),1);
   std::fill(b.begin(),b.end(),2);
   std::fill(c.begin(),c.end(),0);
   a.print();
   b.print();
   c.print();
   int blockSize = 1024;
   int blocks = (a.size()+blockSize-1)/blockSize;
   gpu_add<<<blocks,blockSize>>>(a,b,c);
   cudaDeviceSynchronize();
   a.print();
   b.print();
   c.print();
   for (size_t i = 0 ; i< c.size();i++){
      if (c.at(i)!=3) return false;
   }
   return true;
}

bool gpu_add_double(){
   const int N=1e8;
   //declare 3 vectors 
   dvec a(N);
   dvec b(N);
   dvec c(N);
   std::fill(a.begin(),a.end(),1);
   std::fill(b.begin(),b.end(),2);
   std::fill(c.begin(),c.end(),0);
   a.print();
   b.print();
   c.print();
   int blockSize = 1024;
   int blocks = (a.size()+blockSize-1)/blockSize;
   gpu_add_double<<<blocks,blockSize>>>(a,b,c);
   cudaDeviceSynchronize();
   a.print();
   b.print();
   c.print();
   for (size_t i = 0 ; i< c.size();i++){
      if (c.at(i)!=3) return false;
   }
   return true;
}
bool access(){

   const int N= 1e8;
   vec a(N);
   try{
      a.optimizeCPU();
      a.optimizeGPU();
      a.optimizeCPU();
      int val=0;
      for (size_t i=0; i<a.size(); i++){
         val = a[i];
         val+= a.at(i);
      }  
   }catch(...){
      return false;
   }
   return true;
}


bool operators(){

   const int N= 1e8;
   {
      vec a(N);
      vec b(N);
      std::fill(a.begin(),a.end(),1);
      std::fill(b.begin(),b.end(),2);

      {
         bool same= (a==b); //should be false
         bool different= (a!=b); //should be true
         if (same){return false;}
         if (!different){return false;}
      }

      std::fill(a.begin(),a.end(),1);
      std::fill(b.begin(),b.end(),1);

      {
         bool same = (a==b); //should be true
         bool different= (a!=b); //should be false
         if (!same){return false;}
         if (different){return false;}
      }


      vec c(N);
      vec d(2*N);
      std::fill(c.begin(),c.end(),1);
      std::fill(d.begin(),d.end(),1);


      {
         bool same= (c==d); //should be false
         bool different= (c!=d); //should be true
         if (same){return false;}
         if (!different){return false;}
      }
   }



   vec a(N);
   vec b(N);
   std::fill(a.begin(),a.end(),1);
   std::fill(b.begin(),b.end(),2);
   
   {
      vec c= a+b;
      for (size_t i = 0 ; i< c.size();i++){
         if (c.at(i)!=3) return false;
      }
   }

   {
      vec c= a-b;
      for (size_t i = 0 ; i< c.size();i++){
         if (c.at(i)!=-1) return false;
      }
   }

   {
      vec c= a*b;
      for (size_t i = 0 ; i< c.size();i++){
         if (c.at(i)!=2) return false;
      }
   }

   {
      vec c= b/a;
      for (size_t i = 0 ; i< c.size();i++){
         if (c.at(i)!=2) return false;
      }
   }
   return true;
}


bool ranges_test(){

   const int N= 10;
   vec a(N);
   std::fill(a.begin(),a.end(),10);
   
   int cnt=0;
   for (auto &i:a ){
      i=cnt;
      cnt++;
   }

   for (const auto &i:a ){
      std::cout<<i<<std::endl;
   }


   return true;
}


void perform_tests(){
    IS_TRUE(swap_same_size());
    IS_TRUE(swap_different_size());
#ifdef CUDAVEC
    IS_TRUE(gpu_add());
    IS_TRUE(gpu_add_double());
#endif
    IS_TRUE(access());
    IS_TRUE(operators());
    IS_TRUE(ranges_test());
    std::cout << "\033[1;32m ==========> All OK <========== \033[0m\n"<<std::endl; \
}


int main(){
   cudaProfilerStart();
   perform_tests();
   cudaProfilerStop();
}

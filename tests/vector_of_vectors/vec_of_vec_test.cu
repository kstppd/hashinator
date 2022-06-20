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


bool vec_of_vec(){
   split::SplitVector<vec> b(2); 

   b[0]=vec(10,1);
   b[1]=vec(20,2);
   b[0].print();
   b[1].print();
   b[1].resize(100);
   b[0].print();
   b[1].print();

   //vec a(100);
   //a.print();
   //a.resize(1000);
   //a.print();
   

   return true;
}



void perform_tests(){
   IS_TRUE(vec_of_vec());
    std::cout << "\033[1;32m ==========> All OK <========== \033[0m\n"<<std::endl; \
}


int main(){
   cudaProfilerStart();
   perform_tests();
   cudaProfilerStop();
}

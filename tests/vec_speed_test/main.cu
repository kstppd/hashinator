#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include "/home/kostis/dev/profiny/Profiny.h"

#define N 1

typedef split::SplitVector<int> splitvector ;
typedef std::vector<int> stdvector ;


__global__ 
void stress_kernel(splitvector*a, splitvector* b, splitvector* c){
   
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index< a->size()){
      c->at(index)=a->at(index)+b->at(index);
   }
}


__global__
void change_if(splitvector*a){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i< a->size()){
      if ( a->operator[](i) >1 ){
         a->at(i)=0;
      }
   }
}

void gpu_stress_test(size_t nelems,int threads){
	PROFINY_SCOPE

   splitvector a(nelems,1);
   splitvector b(nelems,2);
   splitvector c(nelems,0);

   splitvector* d_a=a.upload();
   splitvector* d_b=b.upload();
   splitvector* d_c=c.upload();

   stress_kernel<<<nelems/threads,threads>>>(d_a,d_b,d_c);
   cudaDeviceSynchronize();
   change_if<<<nelems/threads,threads>>>(d_a);
   cudaDeviceSynchronize();
   


   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

}

void stress_test_std(stdvector& vec,const size_t elements){
	PROFINY_SCOPE
   for (int ntimes=0; ntimes<N; ntimes++){
      //Load with pushbacks
      for (size_t i=0; i<elements;i++){
         vec.push_back(i);
      }
      //Read random
      const size_t size=vec.size();
      for (size_t i=0; i<elements;i++){
         vec[i]+=1;
         vec[size-i]+=1;
      }
      //Pop all elements
      for (size_t i=0; i<elements;i++){
         vec.pop_back();
      }
   }
   vec=stdvector();
}

void stress_test_split(splitvector& vec,const size_t elements){
	PROFINY_SCOPE
   for (int ntimes=0; ntimes<N; ntimes++){
      //Load with pushbacks
      for (size_t i=0; i<elements;i++){
         vec.push_back(i);
      }
      //Read random
      const size_t size=vec.size();
      for (size_t i=0; i<elements;i++){
         vec[i]+=1;
         vec[size-i]+=1;
      }
      //Pop all elements
      for (size_t i=0; i<elements;i++){
         vec.pop_back();
      }
   }
   vec=splitvector();
}


int main(int argc, char** argv){

   splitvector a;
   stdvector b;

   
	PROFINY_SCOPE
	profiny::Profiler::setOmitRecursiveCalls(false);
   int threads=32;
   for (int power=10; power<30; power++){
      auto start = std::chrono::high_resolution_clock::now();
      gpu_stress_test(1<<power,threads);
      auto end = std::chrono::high_resolution_clock::now();
      auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      printf("TIME: %.5f Power: %d \n", total_time.count() * 1e-9,power);
   }

}

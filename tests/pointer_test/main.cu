#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include <cuda_profiler_api.h>
#include "../../include/splitvector/split_tools.h"
#define N 1024
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
typedef split::SplitVector<int,split::split_unified_allocator<int>,split::split_unified_allocator<size_t>> vec ;

class Managed {
public:
   void *operator new(size_t len) {
      void *ptr;
      cudaMallocManaged(&ptr, len);
      cudaDeviceSynchronize();
      return ptr;
   }

   void operator delete(void *ptr) {
      cudaDeviceSynchronize();
      cudaFree(ptr);
   }

   void* operator new[] (size_t len) {
      void *ptr;
      cudaMallocManaged(&ptr, len);
      cudaDeviceSynchronize();
      return ptr;
   }

   void operator delete[] (void* ptr) {
      cudaDeviceSynchronize();
      cudaFree(ptr);
   }

};

class TestClass:public Managed{
   public:
   TestClass(){
      a= new vec(1024,128);
   }
   ~TestClass(){
      delete a;
   }
   vec* a;
};

__global__
void printClassVec(vec* a){
   printf("----> %d\n",(int)a->size());
   printf("----> %d\n",(int)a->at(12));
}

TEST(Test_GPU,VectorPrint){
   TestClass* test;
   test=new TestClass();
   printClassVec<<<1,1>>>(test->a);
   cudaDeviceSynchronize();
   delete test;
}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

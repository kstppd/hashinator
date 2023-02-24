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
      //cudaMallocManaged(&ptr, len);
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
      value= new int(132);
      *value=132;
   }
   ~TestClass(){
      delete value;
   }
   int* value;
};

__global__
void printClass(TestClass* t){
   printf("-Value on GPU = %d\n",(int)(*(t->value)));
}

TEST(Test_GPU,VectorPrint){

   TestClass* test;
   test=new TestClass();
   printClass<<<1,1>>>(test);
   cudaDeviceSynchronize();
   delete test;

}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

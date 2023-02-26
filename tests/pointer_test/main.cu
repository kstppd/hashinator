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


__global__
void add_vectors(vec* a , vec* b,vec* c){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index< a->size()){
      c->at(index)=a->at(index)+b->at(index);
   }

}

TEST(Test_GPU,VectorAddition){
   vec a(N,1);
   vec b(N,2);
   vec c(N,0);
   
   vec* d_a=a.upload();
   vec* d_b=b.upload();
   vec* d_c=c.upload();

   add_vectors<<<N,32>>>(d_a,d_b,d_c);
   cudaDeviceSynchronize();
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);


   for (const auto& e:c){
      expect_true(e==3);
   }
}


TEST(Test_GPU,VectorAddition2){
   vec* a;
   vec* b;
   vec* c;

   a=new vec(N,1);
   b=new vec(N,2);
   c=new vec(N,0);

   vec* d_a=a->upload();
   vec* d_b=b->upload();
   vec* d_c=c->upload();

   add_vectors<<<1,N>>>(d_a,d_b,d_c);
   cudaDeviceSynchronize();
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);


   for (const auto& e:*c){
      expect_true(e==3);
   }
   delete a;
   delete b;
   delete c;
}

TEST(Test_GPU,VectorAddition3){
   vec* a;
   vec* b;
   vec* c;

   cudaMallocManaged(&a, sizeof(vec));
   cudaMallocManaged(&b, sizeof(vec));
   cudaMallocManaged(&c, sizeof(vec));

   ::new(a) vec(N,1);
   ::new(b) vec(N,2);
   ::new(c) vec(N,0);


   add_vectors<<<1,N>>>(a,b,c);
   cudaDeviceSynchronize();

   for (const auto& e:*c){
      expect_true(e==3);
   }

   a->~vec();
   b->~vec();
   c->~vec();
   cudaFree(a);
   cudaFree(b);
   cudaFree(c);
}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

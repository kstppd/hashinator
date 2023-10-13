#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#define expect_true EXPECT_TRUE

using vec_type_t = int;
using vector =  split::SplitDeviceVector<vec_type_t> ;


void printVecStats(vector* v){
   std::cout<<v->size()<<std::endl;
   std::cout<<v->capacity()<<std::endl;
   std::cout<<"-------------\n";
}


void printVecElements(vector* v){
   std::cout<<v->size()<<std::endl;
   std::cout<<v->capacity()<<std::endl;
   for (size_t i =0;i<v->size();i++){
      std::cout<<v->get(i)<<", ";
   }
   std::cout<<"\n-------------\n";
}


TEST(SplitDeviceVector,Construction){
   
   constexpr size_t N=1<<10;
   vector* a=new vector;
   expect_true(a->size()==0);
   expect_true(a->capacity()==0);

   vector* b=new vector(N);
   expect_true(b->size()==N);
   expect_true(b->capacity()==N);
   
   vector*c =new vector(N);
   vector*d =new vector(*c);
   expect_true(d->size()==c->size());


   std::vector<vec_type_t> s{1,2,3,4,5};
   vector* e = new vector(s);
   expect_true(e->size()==s.size());

   
   split::SplitVector<vec_type_t> k{1,2,3,4,5} ;
   vector* f = new vector(k);
   expect_true(f->size()==k.size());
   delete a;
   delete b;
   delete c;
   delete d;
   delete e;
   delete f;
}

TEST(SplitDeviceVector,SizeModifiers){

   constexpr size_t N=(1<<10);
   vector* a = new vector;
   expect_true(a->size()==0);
   expect_true(a->capacity()==0);
   a->reserve(N);
   expect_true(a->size()==0);
   expect_true(a->capacity()>=N);
   auto cap =a->capacity();
   a->resize(N);
   expect_true(a->size()==N);
   expect_true(a->capacity()==cap);
   delete a;
}

TEST(SplitDeviceVector,HostPushBack){
   
   constexpr size_t N=(1<<10);
   vector* a=new vector;
   for (size_t i =0;i<N;i++){
      a->push_back(i);
   }

   for (size_t i =0;i<N;i++){
      expect_true(a->get(i)==i);
   }

   vector* b=new vector;
   for (size_t i =0;i<N;i++){
      b->push_back(i);
   }

   for (size_t i =0;i<N;i++){
      expect_true(b->get(i)==i);
   }

   vector* c=new vector;
   for (size_t i =0;i<10*N;i++){
      c->push_back(i);
   }

   for (size_t i =0;i<10*N;i++){
      expect_true(c->get(i)==i);
   }
   delete a;
   delete b;
   delete c;
}

__global__
void kernel_set(vector* a){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   a->device_set(index,index);
}

__global__
void kernel_pushback(vector* a){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   a->device_push_back(index);
}

TEST(SplitDeviceVector,DeviceSet){
   
   constexpr size_t N=(1<<10);
   vector* a=new vector;
   a->resize(N);
   kernel_set<<<1,N>>>(a);
   split_gpuDeviceSynchronize();

   for (size_t i =0;i<N;i++){
      expect_true(a->get(i)==i);
   }
   delete a;
}

TEST(SplitDeviceVector,DevicePushBack){
   
   constexpr size_t N=(1<<10);
   vector* a=new vector;
   a->reserve(10*N);
   kernel_pushback<<<10,N>>>(a);
   split_gpuDeviceSynchronize();
   expect_true(a->size()==10*N);
   delete a;
}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

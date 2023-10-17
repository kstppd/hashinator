#include <iostream>
#include <stdlib.h>
#include <vector>
#include <random>
#include <chrono>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"
#define expect_true EXPECT_TRUE

using vec_type_t = int;
using vector =  split::SplitDeviceVector<vec_type_t> ;


void fill_vec(vector* v, size_t targetSize){
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<vec_type_t> dist(1, std::numeric_limits<vec_type_t>::max());
   while (v->size() < targetSize) {
      vec_type_t val =dist(gen);
      v->push_back(val);
    }
}

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

TEST(SplitDeviceVector,AssignmentOperator){
   
   constexpr size_t N=1<<10;
   vector* a=new vector;
   expect_true(a->size()==0);
   expect_true(a->capacity()==0);

   vector* b=new vector;
   *b=*a;
   expect_true(b->size()==0);
   expect_true(b->capacity()==0);

   vector*c =new vector(N);
   c->reserve(10*N);
   vector*d =new vector;
   *d=*c;
   
   expect_true(d->size()==c->size());

   for (size_t i =0;i<d->size();i++){
      expect_true(d->get(i)==c->get(i));
   }
   expect_true((*d==*c));
   expect_true(!(*d!=*c));
   delete a;
   delete b;
   delete c;
   delete d;
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

__global__
void kernel_read_iterators(vector* a){
   for ( auto i=a->device_begin(); i!=a->device_end();++i){
      *i=(*i)*2;
   }
   for ( auto i=a->device_begin(); i!=a->device_end();++i){
   }
}

TEST(SplitDeviceVector,DeviceIterator){
   constexpr size_t N=32;
   vector* a=new vector;
   a->reserve(N);
   kernel_pushback<<<1,N>>>(a);
   split_gpuDeviceSynchronize();
   kernel_read_iterators<<<1,1>>>(a);
   split_gpuDeviceSynchronize();
   delete a;
}

TEST(SplitDeviceVector,HostIterator){
   constexpr size_t N=32;
   vector* a=new vector;
   a->reserve(N);
   kernel_pushback<<<1,N>>>(a);
   split_gpuDeviceSynchronize();
   for (auto i= a->begin(); i!=a->end();++i){
   }
   delete a;
}

bool run_compcation_test(size_t sz){
   vector* v=new vector;
   fill_vec(v,sz);
   auto predicate_on =[]__host__ __device__ (vec_type_t element)->bool{ return element%2 == 0 ;};
   auto predicate_off =[]__host__ __device__ (vec_type_t element)->bool{ return element%2 != 0 ;};
   vector* output1 = new vector(v->size());
   vector* output2 = new vector(v->size());
   const size_t len1 = split::tools::copy_if(v->data(),output1->data(),v->size(),predicate_on);
   const size_t len2 = split::tools::copy_if(v->data(),output2->data(),v->size(),predicate_off);
   auto r=v->size();
   delete v;
   delete output1;
   delete output2;
   return len1+len2==r;
}

TEST(SplitDeviceVector,StreamCompaction){
   for (size_t i = 100; i< 50000; i*=4){
      expect_true(run_compcation_test(i));
   }
}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

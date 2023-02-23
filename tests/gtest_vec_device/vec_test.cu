#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include <cuda_profiler_api.h>
#include "../../include/splitvector/split_tools.h"

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define N 1<<12
typedef split::SplitVector<int,split::split_unified_allocator<int>,split::split_unified_allocator<size_t>> vec ;


__global__
void add_vectors(vec* a , vec* b,vec* c){

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index< a->size()){
      c->at(index)=a->at(index)+b->at(index);
   }

}

__global__
void push_back_kernel(vec* a){

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   a->device_push_back(index);
}

__global__
void merge_kernel(vec* a,vec *b ){

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index==0){
      a->device_insert(a->end(),b->begin(),b->end());
   }
}

__global__
void erase_kernel(vec* a){
   auto it=a->begin();
   a->erase(it);

}



void print_vec_elements(const vec& v){
   std::cout<<"****Vector Contents********"<<std::endl;
   std::cout<<"Size= "<<v.size()<<std::endl;
   std::cout<<"Capacity= "<<v.capacity()<<std::endl;
   for (const auto i:v){
      std::cout<<i<<" ";
   }

   std::cout<<"\n****~Vector Contents********"<<std::endl;
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

TEST(Constructors,Default){
   vec a;
   expect_true(a.size()==0 && a.capacity()==0);
   expect_true(a.data()==nullptr);
}

TEST(Constructors,Size_based){
   vec a(N);
   expect_true(a.size()==N && a.capacity()==N);
   expect_true(a.data()!=nullptr);
}


TEST(Constructors,Specific_Value){
   vec a(N,5);
   expect_true(a.size()==N && a.capacity()==N);
   for (size_t i=0; i<N;i++){
      expect_true(a[i]==5);
      expect_true(a.at(i)==5);
   }
}

TEST(Constructors,Copy){
   vec a(N,5);
   vec b(a);
   for (size_t i=0; i<N;i++){
      expect_true(a[i]==b[i]);
      expect_true(a.at(i)==b.at(i));
   }
}

TEST(Vector_Functionality , Reserve){
   vec a;
   size_t cap =1000000;
   a.reserve(cap);
   expect_true(a.size()==0);
   expect_true(a.capacity()==cap);
}

TEST(Vector_Functionality , Resize){
   vec a;
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   expect_true(a.capacity()==a.size());
}

TEST(Vector_Functionality , Swap){
   vec a(10,2);
   vec b(10,2);
   a.swap(b);
   vec c(100,1);
   vec d (200,3);
   c.swap(d);
   expect_true(c.size()==200);
   expect_true(d.size()==100);
   expect_true(c.front()==3);
   expect_true(d.front()==1);

}

TEST(Vector_Functionality , Resize2){
   vec a;
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   expect_true(a.capacity()==a.size());
}

TEST(Vector_Functionality , Clear){
   vec a(10);
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   auto cap=a.capacity();
   a.clear();
   expect_true(a.size()==0);
   expect_true(a.capacity()==cap);
}

TEST(Vector_Functionality , PopBack){
   vec a{1,2,3,4,5,6,7,8,9,10};
   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();
   for (int i=9;i>=0;i--){
      a.pop_back();
      if (a.size()>0){
         expect_true(i==a.back());
      }
   }
   expect_true(a.size()==0);
   expect_true(a.capacity()==initial_cap);
}

TEST(Vector_Functionality , Push_Back){
   vec a;
   for (auto i=a.begin(); i!=a.end();i++){
      expect_true(false);
   }

   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   a.push_back(11);
   expect_true(11==a[a.size()-1]);
   a.push_back(12);
   expect_true(12==a[a.size()-1]);

}


TEST(Vector_Functionality , Shrink_to_Fit){
   vec a;
   for (auto i=a.begin(); i!=a.end();i++){
      expect_true(false);
   }

   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   for (int i =0 ; i< 1024; i++){
      a.push_back(i);
   }

   expect_true(a.size()<a.capacity());
   a.shrink_to_fit();
   expect_true(a.size()==a.capacity());

}
TEST(Vector_Functionality , Push_Back_2){
   vec a{1,2,3,4,5,6,7,8,9,10};
   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   a.push_back(11);
   expect_true(11==a[a.size()-1]);
   a.push_back(12);
   expect_true(12==a[a.size()-1]);

}

TEST(Vector_Functionality , Insert_1_Element){
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      auto it(a.begin());
      auto it2=a.insert(it,-1);
      expect_true(a[0]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(a.end());
      auto it2=a.insert(it,-1);
      expect_true(a.back()=-1);
      expect_true(a[a.size()-1]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(&a[4]);
      auto it2=a.insert(it,-1);
      expect_true(a[4]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
     vec a{1,2,3,4,5,6,7,8,9,10};
     auto s0=a.size(); auto c0=a.capacity();
     try {
      //hehe
      vec::iterator it(nullptr);
      auto it2=a.insert(it,-1);
     }// this has to throw
     catch (...) {
        expect_true(true);
        expect_true(a.capacity()==c0);
        expect_true(a.size()==s0);
           return;
     }
     //if we end up here it never threw so something's up
     expect_true(false);
   }
}


TEST(Vector_Functionality , Insert_Many_Elements){

   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      auto it(a.begin());
      auto it2=a.insert(it,10,-1);
      for (int i =0 ; i<10 ; i++){
         expect_true(a[i]=-1);
      }
      expect_true(a.front()==-1);
      expect_true(a.size()==s0+10);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(a.end());
      auto it2=a.insert(it,10,-1);
      for (int i =s0 ; i<a.size() ; i++){
         expect_true(a[i]=-1);
      }
      expect_true(a.back()=-1);
      expect_true(a.size()==s0+10);
   }

   {
     vec a{1,2,3,4,5,6,7,8,9,10};
     auto s0=a.size(); auto c0=a.capacity();
     try {
      //hehe
      vec::iterator it(nullptr);
      auto it2=a.insert(it,10,-1);
     }// this has to throw
     catch (...) {
        expect_true(true);
        expect_true(a.capacity()==c0);
        expect_true(a.size()==s0);
           return;
     }
     //if we end up here it never threw so something's up
     expect_true(false);
   }
}


TEST(Vector_Functionality , Insert_Range_Based){

   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto backup(a);
      vec b{-1,-2,-3,-4,-5,-6,-7,-8,-9,-10};
      auto s0=a.size();
      auto it(a.end());
      auto it_b0(b.begin());
      auto it_b1(b.end());
      a.insert(it,it_b0,it_b1);
      expect_true(a.size()==s0+b.size());
      for (int i=0 ; i <10 ; i++){
         expect_true(a[i]=backup[i]);

      }
      for (int i=10 ; i <20 ; i++){
         expect_true(a[i]=b[i-10]);
      }
   }


   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto backup(a);
      vec b{-1,-2,-3,-4,-5,-6,-7,-8,-9,-10};
      auto s0=a.size();
      auto it(a.end());
      auto it_b0(b.begin());
      auto it_b1(b.begin());
      a.insert(it,it_b0,it_b1);
      expect_true(a.size()==s0);
      for (int i=0 ; i <10 ; i++){
         expect_true(a[i]=backup[i]);

      }
   }

}

TEST(Vector_Functionality , Erase_Single){
      vec a{1,2,3,4,5,6,7,8,9,10};
      vec::iterator it(&a[4]);
      auto backup=*it;
      auto s0=a.size();
      a.erase(it);
      expect_true(backup!=*it);
      expect_true(a.size()==s0-1);
}

TEST(Vector_Functionality , PushBack_And_Erase_Device){
      vec a;
      a.reserve(100);
      vec* d_a=a.upload();
      push_back_kernel<<<4,8>>>(d_a);
      cudaDeviceSynchronize();
      cudaFree(d_a);
      vec* d_b=a.upload();
      erase_kernel<<<1,1>>>(d_b);
      cudaDeviceSynchronize();
      cudaFree(d_b);
}

TEST(Vector_Functionality , Insert_Device){
      vec a{1,2,3,4};
      a.reserve(20);
      vec b{5,6,7,8};
      vec* d_a=a.upload();
      vec* d_b=b.upload();
      merge_kernel<<<1,1>>>(d_a,d_b);
      cudaDeviceSynchronize();
      merge_kernel<<<1,1>>>(d_a,d_b);
      cudaDeviceSynchronize();
      cudaFree(d_a);
      cudaFree(d_b);
}

TEST(Vector_Functionality , Bug){
   const vec blocks{1,2,3,4,5,6,7,8,9};
   vec localToGlobalMap{-1,-1,-1,-1};
   localToGlobalMap.insert(localToGlobalMap.end(),blocks.begin(),blocks.end());
}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

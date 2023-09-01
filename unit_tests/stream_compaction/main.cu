#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define N 1<<8
#define N2 1<<20
#define WARP 32
#define BLOCKSIZE 32

typedef uint32_t val_type;
typedef split::SplitVector<val_type> vec ;



void print_vec_elements(vec& v,const char*name=" "){
   std::cout<<name <<"****Vector Contents********"<<std::endl;
   std::cout<<"Size= "<<v.size()<<std::endl;
   std::cout<<"Capacity= "<<v.capacity()<<std::endl;
   size_t j=0;
   for (auto i:v){
      printf("%d ",i);
   }
   std::cout<<"\n****~Vector Contents********"<<std::endl;
}


struct Predicate{
   __host__ __device__
   bool operator ()(int i)const {
      return i%2==0;
   }
};

template <typename T>
void naive_xScan(const split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                 split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>&  output)
{
	output[0] = 0; //exclusive
	for(size_t i = 0; i < input.size()-1 ; i++) {
		output[i+1] = output[i] + input[i];
	}
}


TEST(Compaction,GPU_CPU){
   const size_t sz=N;
   vec input(sz);
   vec output(sz);
   vec cpu_output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});

   split_tools::copy_if<val_type,Predicate,BLOCKSIZE,WARP>(input,output,Predicate());

   Predicate rule;size_t j =0;
   for (const auto& e:input){
      if (rule(e)){cpu_output[j++]=e;}
   }

   cpu_output.erase(&cpu_output.at(j),cpu_output.end());
   expect_true(output==cpu_output);
}


TEST(PrefixScan,GPU_CPU){
   const size_t sz=N2;
   vec input(sz);
   vec output(sz);
   vec cpu_output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});
   split_tools::split_prefix_scan<val_type,BLOCKSIZE,WARP>(input,output);
   naive_xScan(input,cpu_output);
   expect_true(output==cpu_output);
   std::cout<<"GPU scan = "<<output.back()<<std::endl;
}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

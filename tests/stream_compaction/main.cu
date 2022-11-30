#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>
#include "../../src/splitvector/split_tools.h"

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define N 1<<20
#define N2 1000000
#define WARP 32
#define BLOCKSIZE 32

typedef uint32_t val_type;
typedef split::SplitVector<val_type,split::split_unified_allocator<val_type>,split::split_unified_allocator<size_t>> vec ;

bool isPow2(size_t x ){
   return (x&(x-1))==0;
}


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



void split_prefix_scan(vec& input, vec& output){
   //Input size
   const size_t input_size=input.size();

   //Scan is performed in half Blocksizes
   size_t scanBlocksize= BLOCKSIZE/2;
   size_t scanElements=2*scanBlocksize;
   size_t gridSize = input_size/scanElements;

   //If input is not exactly divisible by scanElements we launch an extra block
   //assert(isPow2(input_size) && "Using prefix scan with non powers of 2 as input size is not thought out yet :D");

   if (input_size%scanElements!=0){
      std::cout<<"Gridsize was"<<gridSize+1<<std::endl;
      gridSize=1<<((int)ceil(log(++gridSize)/log(2)));
      std::cout<<"Gridsize is"<<gridSize<<std::endl;
   }

   //Allocate memory for partial sums
   vec partial_sums(gridSize); 

   split_tools::split_prescan<<<gridSize,scanBlocksize,scanElements*sizeof(val_type)>>>(input.data(),output.data(),partial_sums.data(),scanElements,input.size());
   cudaDeviceSynchronize();


   if (gridSize>1){
      if (partial_sums.size()<scanElements){
         vec partial_sums_dummy(gridSize); 
         split_tools::split_prescan<<<1,scanBlocksize,scanElements*sizeof(val_type)>>>(partial_sums.data(),partial_sums.data(),partial_sums_dummy.data(),gridSize,partial_sums.size());
         cudaDeviceSynchronize();
      }else{
         vec partial_sums_clone(partial_sums);
         split_prefix_scan(partial_sums_clone,partial_sums);
      }
      split_tools::scan_add<<<gridSize,scanBlocksize>>>(output.data(),partial_sums.data(),scanElements,output.size());
      cudaDeviceSynchronize();
   }
   vec gpu_out(output);
   naive_xScan(input,output);
   assert(gpu_out==output && "CPU did not match GPU scan so bailing out");
}


TEST(Compaction,GPU){
   const size_t sz=N;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});
   size_t nBlocks=input.size()/BLOCKSIZE; 
   vec counts(nBlocks);
   vec offsets(nBlocks);
   
   //Step 1 -- Per wrap workload
   split_tools::scan_reduce<<<nBlocks,BLOCKSIZE>>>(input.upload(),counts.upload(),Predicate());
   cudaDeviceSynchronize();

   //Step 2 -- Exclusive Prefix Scan on offsets
   split_prefix_scan(counts,offsets);
   cudaDeviceSynchronize();

   ////Step 3 -- Compaction
   vec* d_input=input.upload();
   vec* d_output=output.upload();
   vec* d_offsets=offsets.upload();
   vec* d_counts=counts.upload();
   split_tools::split_compact<val_type,Predicate,BLOCKSIZE,WARP><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int)>>>(d_input,d_counts,d_offsets,d_output,Predicate());
   cudaDeviceSynchronize();
   std::cout<<"GPU compaction = "<<output.back()<<std::endl;
}


TEST(Compaction,CPU){
   const size_t sz=N;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});
   
   Predicate rule;size_t j =0;
   for (const auto& e:input){
      if (rule(e)){output[j++]=e;}
   }
   output.erase(&output.at(j),output.end());
   std::cout<<"CPU compaction = "<<output.back()<<std::endl;
}

TEST(PrefixScan,CPU){
   const size_t sz=N2;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});
   naive_xScan(input,output);
   std::cout<<"CPU scan = "<<output.back()<<std::endl;
}


TEST(PrefixScan,GPU){
   const size_t sz=N2;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static size_t i=0 ; return i++;});
   split_prefix_scan(input,output);
   std::cout<<"GPU scan = "<<output.back()<<std::endl;
}




int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

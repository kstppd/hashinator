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
#define N 1<<12
#define WARP 32
#define BLOCKSIZE 2048
#define PREFIX_BLOCKSIZE 1024
#define MAX_SIZE_PER_BLOCK 32
#define FULL_MASK 0xffffffff

typedef split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>> vec ;


struct Predicate{
   __host__ __device__
   bool operator ()(int i)const {
      return i%2==0;
   }
};



//TEST(Compaction,BlockCount){

   //const size_t sz=128;
   //vec input(sz);
   //vec output(sz);
   //std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   //int nBlocks=input.size()/BLOCKSIZE; 
   //vec counts(nBlocks);
   //vec offsets(nBlocks);
   //vec* d_in=input.upload();
   //vec* d_out=counts.upload() ;
   
   ////Step 1 -- Per wrap workload
   //warpcount_reduction<<<nBlocks,BLOCKSIZE>>>(d_in,d_out,Predicate());
   //cudaDeviceSynchronize();

   ////Step 2 -- Exclusive Prefix Scan on offsets
   //naive_xScan(counts,offsets);

   ////Step 3 -- Compaction
   //vec* d_input=input.upload();
   //vec* d_output=output.upload();
   //vec* d_offsets=offsets.upload();
   //vec* d_counts=counts.upload();
   //split_compact<<<nBlocks,BLOCKSIZE>>>(d_input,d_counts,d_offsets,d_output,Predicate());
   //cudaDeviceSynchronize();
   //expect_true(1);
//}




void split_prefix_scan(vec& input, vec& output){

   //Calculate how many blocks we need to split the array into.
   assert(input.size()%BLOCKSIZE==0);
   const size_t n = input.size()/BLOCKSIZE;
   vec partial_sums(n);
   
   //First scan the full input into its subparts
   //Partial sums contains the partial sub sums after the call 
   split_tools::split_scan<uint32_t,BLOCKSIZE,PREFIX_BLOCKSIZE><<<n,PREFIX_BLOCKSIZE>>>(input.data(),output.data(),partial_sums.data(),BLOCKSIZE);
   cudaDeviceSynchronize();

   //Scan the partial sums in place this time
   split_tools::split_scan_block<uint32_t,PREFIX_BLOCKSIZE><<<1,PREFIX_BLOCKSIZE>>>(partial_sums.data(),partial_sums.data(),partial_sums.size());
   cudaDeviceSynchronize();

   //Finally add the prefix sums to the output
   split_tools::scan_add<uint32_t,BLOCKSIZE><<<n,PREFIX_BLOCKSIZE>>>(output.data(),partial_sums.data(),BLOCKSIZE);
   cudaDeviceSynchronize();
}


TEST(Compaction_GPU,Exclusive_Scan_Large_Array){

   const size_t bl_size=2048;
   const size_t sz=bl_size*2048;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   split_prefix_scan(input,output);
   std::cout<<output.back()<<std::endl;
}


TEST(Compaction_CPU,Exclusive_Scan_Large_Array){

   const size_t bl_size=2048;
   const size_t sz=bl_size*2048;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   split_tools::cpu_exclusive_scan(input,output);
   std::cout<<output.back()<<std::endl;
}



int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

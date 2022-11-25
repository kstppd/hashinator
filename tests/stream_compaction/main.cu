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
#define N 1<<24
#define WARP 32
#define BLOCKSIZE 1024
#define FULL_MASK 0xffffffff

typedef int val_type;
typedef split::SplitVector<val_type,split::split_unified_allocator<val_type>,split::split_unified_allocator<size_t>> vec ;

void print_vec_elements(vec& v){
   std::cout<<"****Vector Contents********"<<std::endl;
   std::cout<<"Size= "<<v.size()<<std::endl;
   std::cout<<"Capacity= "<<v.capacity()<<std::endl;
   for (auto i:v){
      std::cout<<i<<" ";
   }

   std::cout<<"\n****~Vector Contents********"<<std::endl;
}


struct Predicate{
   __host__ __device__
   bool operator ()(int i)const {
      return i%2==0;
      if (i>0 && i <10 ){return true;}
      if (i>300 && i <350 ){return true;}
      return false;

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

   const int PREFIX_BLOCKSIZE=BLOCKSIZE/2;
   //Calculate how many blocks we need to split the array into.
   assert(input.size()%BLOCKSIZE==0);
   const size_t n = input.size()/BLOCKSIZE;
   vec partial_sums(n);
   
   //First scan the full input into its subparts
   //Partial sums contains the partial sub sums after the call 
   split_tools::split_scan<val_type,BLOCKSIZE,PREFIX_BLOCKSIZE><<<n,PREFIX_BLOCKSIZE>>>(input.data(),output.data(),partial_sums.data(),BLOCKSIZE);
   cudaDeviceSynchronize();

   //Scan the partial sums in place this time
   split_tools::split_scan_block<val_type,PREFIX_BLOCKSIZE><<<1,PREFIX_BLOCKSIZE>>>(partial_sums.data(),partial_sums.data(),partial_sums.size());
   cudaDeviceSynchronize();

   //Finally add the prefix sums to the output
   split_tools::scan_add<val_type,BLOCKSIZE><<<n,PREFIX_BLOCKSIZE>>>(output.data(),partial_sums.data(),BLOCKSIZE);
   cudaDeviceSynchronize();
}




template <typename T, typename Rule>
__global__
void compact(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* counts,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* offsets,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                   Rule rule)
{
   extern __shared__ unsigned int buffer[];
   unsigned int size=input->size();
   unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
   unsigned int wid = tid/WARP;
   unsigned int widb = threadIdx.x/WARP;
   unsigned int w_tid=tid%WARP;
   unsigned int warps_in_block = blockDim.x/WARP;
   bool tres=rule(input->at(tid));

   unsigned int  mask= __ballot_sync(FULL_MASK,tres);
   unsigned int n_neighbors= mask & ((1 << w_tid) - 1);
   unsigned int total_valid_in_warp	= __popc(mask);
   if (w_tid==0 ){
      buffer[widb]=total_valid_in_warp;
   }
   __syncthreads();

   if (w_tid==0 && wid%warps_in_block==0){
      buffer[widb]=0;
      for (unsigned int i =1 ; i < warps_in_block;++i){
         buffer[widb+i]+=buffer[widb+i-1];
      }
   }
   __syncthreads();
   unsigned int private_index	= buffer[widb] + offsets->at(wid/warps_in_block) + __popc(n_neighbors);
   //Maybe add an interim step where you push these into a shared memory block first
   if (tres){
      output->at(private_index) = input->at(tid);
   }
   if (tid==0){
      unsigned int actual_total_blocks=offsets->back()+counts->back();
      output->erase(&output->at(actual_total_blocks),output->end());
   }
}

TEST(Compaction,GPU){
   const size_t sz=N;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   int nBlocks=input.size()/BLOCKSIZE; 
   vec counts(nBlocks);
   vec offsets(nBlocks);
   
   //Step 1 -- Per wrap workload
   split_tools::warpcount_reduction<<<nBlocks,BLOCKSIZE>>>(input.upload(),counts.upload(),Predicate());
   cudaDeviceSynchronize();

   //Step 2 -- Exclusive Prefix Scan on offsets
   //split_prefix_scan(counts,offsets);
   //cudaDeviceSynchronize();
   naive_xScan(counts,offsets);

   ////Step 3 -- Compaction
   vec* d_input=input.upload();
   vec* d_output=output.upload();
   vec* d_offsets=offsets.upload();
   vec* d_counts=counts.upload();
   compact<val_type,Predicate><<<nBlocks,BLOCKSIZE,(BLOCKSIZE/WARP)*sizeof(unsigned int)>>>(d_input,d_counts,d_offsets,d_output,Predicate());
   cudaDeviceSynchronize();
   std::cout<<"GPU compaction = "<<output.back()<<std::endl;
}


TEST(Compaction,CPU){
   const size_t sz=N;
   vec input(sz);
   vec output(sz);
   std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   
   Predicate rule;size_t j =0;
   for (const auto& e:input){
      if (rule(e)){output[j++]=e;}
   }
   output.erase(&output.at(j),output.end());
   std::cout<<"CPU compaction = "<<output.back()<<std::endl;
}



//TEST(Compaction_GPU,Exclusive_Scan_Large_Array){

   //const size_t bl_size=1;
   //const size_t sz=bl_size*64;
   //vec input(sz);
   //vec output(sz);
   //std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   //split_prefix_scan(input,output);
   //std::cout<<output.back()<<std::endl;
//}


//TEST(Compaction_CPU,Exclusive_Scan_Large_Array){

   //const size_t bl_size=1;
   //const size_t sz=bl_size*64;
   //vec input(sz);
   //vec output(sz);
   //std::generate(input.begin(), input.end(), []{static int i=0 ; return i++;});
   //split_tools::cpu_exclusive_scan(input,output);
   //std::cout<<output.back()<<std::endl;
//}



int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

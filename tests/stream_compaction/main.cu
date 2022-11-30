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
#define N2 1025
#define WARP 32
#define BLOCKSIZE 1024
#define FULL_MASK 0xFFFFFFFF //32-bit wide for cuda warps

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
      //printf("[%d,%d] ",j++,i);
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
   if (tid>=size) {return;}
   unsigned int offset = BLOCKSIZE/WARP;
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
      buffer[offset+widb]=0;
      for (unsigned int i =0 ; i < warps_in_block-1;++i){
         buffer[offset+widb+i+1]=buffer[offset+widb+i] +buffer[widb+i];
      }
   }
   __syncthreads();
   const unsigned int neighbor_count= __popc(n_neighbors);
   unsigned int private_index	= buffer[offset+widb] + offsets->at(wid/warps_in_block) + neighbor_count ;
   if (tres && widb!=warps_in_block){
      output->at(private_index) = input->at(tid);
   }
   __syncthreads();
   if (tid==0){
      unsigned int actual_total_blocks=offsets->back()+counts->back();
      output->erase(&output->at(actual_total_blocks),output->end());
   }
}

template <typename T>
__global__
void split_prescan(T* input,T* output,T* partial_sums, size_t n,size_t len){

   extern __shared__ T buffer[];
   int tid = threadIdx.x;
   int offset=1;
   size_t local_start=blockIdx.x*n;

   //Load into shared memory 
   if (tid<len){
      buffer[2*tid]= input[local_start+ 2*tid];
      buffer[2*tid+1]= input[local_start+ 2*tid+1];
   }

   //Reduce Phase
   for (int d =n>>1; d>0; d>>=1){
      __syncthreads();

      if (tid<d){
         int ai = offset*(2*tid+1)-1;
         int bi = offset*(2*tid+2)-1;
         buffer[bi]+=buffer[ai];
      }
      offset*=2;
   }
   
   //Exclusive scan so zero out last element (will propagate to first)
   if (tid==0){ partial_sums[blockIdx.x]=buffer[n-1] ;buffer[n-1]=0;}

   //Downsweep Phase
   for (int d =1; d<n; d*=2){

      offset>>=1;
      __syncthreads();
      if (tid<d){
         int ai = offset*(2*tid+1)-1;
         int bi = offset*(2*tid+2)-1;
         T tmp=buffer[ai];
         buffer[ai]=buffer[bi];
         buffer[bi]+=tmp;
      }
   }
   __syncthreads();
   if (tid<len){
      output[local_start+2*tid]   = buffer[2*tid];
      output[local_start+2*tid+1]= buffer[2*tid+1];
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
   if (input_size%scanElements!=0){gridSize+=1;}

   //Allocate memory for partial sums
   vec partial_sums(gridSize); 

   split_prescan<<<gridSize,scanBlocksize,scanElements*sizeof(val_type)>>>(input.data(),output.data(),partial_sums.data(),scanElements,input.size()/2);
   cudaDeviceSynchronize();

   if (gridSize>1){
      if (partial_sums.size()<scanElements){
         vec partial_sums_dummy(gridSize); 
         split_prescan<<<1,scanBlocksize,scanElements*sizeof(val_type)>>>(partial_sums.data(),partial_sums.data(),partial_sums_dummy.data(),gridSize,partial_sums.size()/2);
         cudaDeviceSynchronize();
      }else{
         vec partial_sums_clone(partial_sums);
         split_prefix_scan(partial_sums_clone,partial_sums);
      }
      split_tools::scan_add<<<gridSize,scanBlocksize>>>(output.data(),partial_sums.data(),scanElements);
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
   split_tools::warpcount_reduction<<<nBlocks,BLOCKSIZE>>>(input.upload(),counts.upload(),Predicate());
   cudaDeviceSynchronize();

   //Step 2 -- Exclusive Prefix Scan on offsets
   split_prefix_scan(counts,offsets);
   cudaDeviceSynchronize();

   ////Step 3 -- Compaction
   vec* d_input=input.upload();
   vec* d_output=output.upload();
   vec* d_offsets=offsets.upload();
   vec* d_counts=counts.upload();
   compact<val_type,Predicate><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int)>>>(d_input,d_counts,d_offsets,d_output,Predicate());
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

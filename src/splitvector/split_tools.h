#pragma once 
#include "split_allocators.h"
#define FULL_MASK 0xFFFFFFFF //32-bit wide for cuda warps not sure what we do on AMD HW

namespace split_tools{

   template <typename T>
   __global__
   void scan_add(T* input,T* partial_sums, size_t blockSize,size_t len){
      const T val=partial_sums[blockIdx.x];
      const size_t target1 = 2*blockIdx.x*blockDim.x+threadIdx.x;
      const size_t target2 = target1+blockDim.x;
      if (target1<len){
         input[target1]+=val;
         if (target2<len){
            input[target2]+=val;
         }
      }
   }

   template<typename T,typename Rule>
   __global__
   void scan_reduce(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                   Rule rule){

      size_t size=input->size();
      size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid<size){
         size_t total_valid_elements=__syncthreads_count(rule(input->at(tid)));
         if (threadIdx.x==0){
            output->at(blockIdx.x)=total_valid_elements;
         }
      }
   }

   template <typename T>
   __global__
   void split_prescan(T* input,T* output,T* partial_sums, int n,size_t len){

      extern __shared__ T buffer[];
      size_t tid = threadIdx.x;
      size_t offset=1;
      size_t local_start=blockIdx.x*n;

      //Load into shared memory 
      if (local_start+2*tid+1<len){
         buffer[2*tid]= input[local_start+ 2*tid];
         buffer[2*tid+1]= input[local_start+ 2*tid+1];
      }

      //Reduce Phase
      for (size_t d =n>>1; d>0; d>>=1){
         __syncthreads();

         if (tid<d){
            size_t ai = offset*(2*tid+1)-1;
            size_t bi = offset*(2*tid+2)-1;
            buffer[bi]+=buffer[ai];
         }
         offset*=2;
      }
      
      //Exclusive scan so zero out last element (will propagate to first)
      if (tid==0){ partial_sums[blockIdx.x]=buffer[n-1] ;buffer[n-1]=0;}

      //Downsweep Phase
      for (size_t d =1; d<n; d*=2){

         offset>>=1;
         __syncthreads();
         if (tid<d){
            size_t ai = offset*(2*tid+1)-1;
            size_t bi = offset*(2*tid+2)-1;
            T tmp=buffer[ai];
            buffer[ai]=buffer[bi];
            buffer[bi]+=tmp;
         }
      }
      __syncthreads();
      if (local_start+2*tid+1<len){
         output[local_start+2*tid]   = buffer[2*tid];
         output[local_start+2*tid+1]= buffer[2*tid+1];
      }
   }

   template <typename T,typename Rule,size_t BLOCKSIZE=1024, size_t WARP=32>
   __global__
   void split_compact(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* counts,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* offsets,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                      Rule rule)
   {
      extern __shared__ T buffer[];
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
}









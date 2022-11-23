#pragma once 
#include "split_allocators.h"


namespace split_tools{

   template <typename T>
   void cpu_exclusive_scan(const split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                    split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>&  output){
      output[0] = 0; //exclusive
      for(size_t i = 0; i < input.size()-1 ; i++) {
         output[i+1] = output[i] + input[i];
      }
   }


   template <typename T, int BLOCKSIZE>
   __global__
   void scan_add(T* input,T* partial_sums, size_t n){
      size_t tid = threadIdx.x;
      size_t local_start=(blockIdx.x)*BLOCKSIZE;
      T val=partial_sums[blockIdx.x];
      input[local_start+2*tid]+=val;
      input[local_start+2*tid+1]+=val;
   }


   template <typename T, typename Rule, int FULL_MASK, int WARP>
   __global__
   void split_compact(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* counts,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* offsets,
                      split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                      Rule rule)
   {

      //extern __shared__ T buffer[];
      size_t size=input->size();
      size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
      size_t wid = tid/WARP;
      size_t w_tid=tid%WARP;
      bool tres=rule(input->at(tid));
      if (tid<input->size()){
         unsigned int  mask= __ballot_sync(FULL_MASK,tres);
         unsigned int n_neighbors= mask & ((1 << w_tid) - 1);
         int total_valid_in_warp	= __popc(mask);
         int private_index	= offsets->at(wid) + __popc(n_neighbors);

         //TODO
         //Maybe add an interim step where you push these into a shared memory block first
         if (tres){
            output->at(private_index) = input->at(tid);
         }
         
         if (tid==0){
            size_t actual_total_blocks=offsets->back()+counts->back();
            output->erase(&output->at(actual_total_blocks),output->end());
         }
      }
   }

   template <typename T, int PREFIX_BLOCKSIZE >
   __global__
   void split_scan_block(T* input,T* output,size_t n){

      __shared__ T buffer[2*PREFIX_BLOCKSIZE];
      size_t tid = threadIdx.x;
      int offset=1;

      //Load into shared memory 
      buffer[2*tid]= input[2*tid];
      buffer[2*tid+1]= input[2*tid+1];

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
      if (tid==0){ buffer[n-1]=0;}

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
      output[2*tid]   = buffer[2*tid];
      output[2*tid+1]= buffer[2*tid+1];
   }


   template <typename T,int BLOCKSIZE, int PREFIX_BLOCKSIZE>
   __global__
   void split_scan(T* input,T* output,T* partial_sums, size_t n){

      __shared__ T buffer[2*PREFIX_BLOCKSIZE];
      size_t tid = threadIdx.x;
      int offset=1;
      size_t local_start=blockIdx.x*BLOCKSIZE;


      //Load into shared memory 
      buffer[2*tid]= input[local_start+ 2*tid];
      buffer[2*tid+1]= input[local_start+ 2*tid+1];

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
      output[local_start+2*tid]   = buffer[2*tid];
      output[local_start+2*tid+1]= buffer[2*tid+1];
   }


   template<typename T,typename Rule>
   __global__
   void warpcount_reduction(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                   Rule rule){

      size_t size=input->size();
      size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid<size){
         int total_valid_elements=__syncthreads_count(rule(input->at(tid)));
         if (threadIdx.x==0){
            output->at(blockIdx.x)=total_valid_elements;
         }
      }
   }
}









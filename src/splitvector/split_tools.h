#pragma once 
#include "split_allocators.h"
#define SPLIT_VOTING_MASK 0xFFFFFFFF //32-bit wide for cuda warps not sure what we do on AMD HW

namespace split_tools{

   bool isPow2(const size_t val ) {
      return (val &(val-1))==0;
   }

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
      int tid = threadIdx.x;
      int offset=1;
      int local_start=blockIdx.x*n;

      //Load into shared memory 
      if (local_start+2*tid+1<len){
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
      const size_t size=input->size();
      const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid>=size) {return;}
      unsigned int offset = BLOCKSIZE/WARP;
      const unsigned int wid = tid/WARP;
      const unsigned int widb = threadIdx.x/WARP;
      const unsigned int w_tid=tid%WARP;
      const unsigned int warps_in_block = blockDim.x/WARP;
      const bool tres=rule(input->at(tid));

      unsigned int  mask= __ballot_sync(SPLIT_VOTING_MASK,tres);
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
      const unsigned int private_index	= buffer[offset+widb] + offsets->at(wid/warps_in_block) + neighbor_count ;
      if (tres && widb!=warps_in_block){
         output->at(private_index) = input->at(tid);
      }
      __syncthreads();
      if (tid==0){
         const unsigned int actual_total_blocks=offsets->back()+counts->back();
         output->erase(&output->at(actual_total_blocks),output->end());
      }
   }


   template <typename T, size_t BLOCKSIZE=1024,size_t WARP=32>
   void split_prefix_scan(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                          split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& output )

   {
      using vector=split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>;

      //Input size
      const size_t input_size=input.size();

      //Scan is performed in half Blocksizes
      size_t scanBlocksize= BLOCKSIZE/2;
      size_t scanElements=2*scanBlocksize;
      size_t gridSize = input_size/scanElements;

      //If input is not exactly divisible by scanElements we launch an extra block
      assert(isPow2(input_size) && "Using prefix scan with non powers of 2 as input size is not thought out yet :D");

      if (input_size%scanElements!=0){
         gridSize=1<<((int)ceil(log(++gridSize)/log(2)));
      }

      //Allocate memory for partial sums
      vector partial_sums(gridSize); 

      split_tools::split_prescan<<<gridSize,scanBlocksize,scanElements*sizeof(T)>>>(input.data(),output.data(),partial_sums.data(),scanElements,input.size());
      cudaDeviceSynchronize();


      if (gridSize>1){
         if (partial_sums.size()<scanElements){
            vector partial_sums_dummy(gridSize); 
            split_tools::split_prescan<<<1,scanBlocksize,scanElements*sizeof(T)>>>(partial_sums.data(),partial_sums.data(),partial_sums_dummy.data(),gridSize,partial_sums.size());
            cudaDeviceSynchronize();
         }else{
            vector partial_sums_clone(partial_sums);
            split_prefix_scan(partial_sums_clone,partial_sums);
         }
         split_tools::scan_add<<<gridSize,scanBlocksize>>>(output.data(),partial_sums.data(),scanElements,output.size());
         cudaDeviceSynchronize();
      }
   }

   
   template <typename T, typename Rule,size_t BLOCKSIZE=1024,size_t WARP=32>
   void copy_if(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& output,
                Rule rule)
   {


      using vector=split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>;
      
      //Figure out Blocks to use
      size_t nBlocks=input.size()/BLOCKSIZE; 
      vector counts(nBlocks);
      vector offsets(nBlocks);

            
      //Phase 1 -- Calculate per warp workload
      vector * d_input=input.upload();
      vector * d_counts=counts.upload();
      split_tools::scan_reduce<<<nBlocks,BLOCKSIZE>>>(d_input,d_counts,rule);
      cudaDeviceSynchronize();
      cudaFree(d_input);
      cudaFree(d_counts);


      //Step 2 -- Exclusive Prefix Scan on offsets
      split_prefix_scan<T,BLOCKSIZE,WARP>(counts,offsets);
      cudaDeviceSynchronize();


      //Step 3 -- Compaction
      vector* d_output=output.upload();
      vector* d_offsets=offsets.upload();
      d_input=input.upload();
      d_counts=counts.upload();
      split_tools::split_compact<T,Rule,BLOCKSIZE,WARP><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int)>>>(d_input,d_counts,d_offsets,d_output,rule);
      cudaDeviceSynchronize();

      //Deallocate the handle pointers
      cudaFree(d_input);
      cudaFree(d_counts);
      cudaFree(d_output);
      cudaFree(d_offsets);
   }
}

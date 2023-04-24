/* File:    split::tools.h
 * Authors: Kostis Papadakis (2023)
 * Description: Set of tools used by SplitVectors
 *
 * This file defines the following classes or functions:
 *    --split::tools::Cuda_mempool
 *    --split::tools::copy_if_raw
 *    --split::tools::copy_if
 *    --split::tools::scan_reduce_raw
 *    --split::tools::scan_reduce
 *    --split::tools::scan_add
 *    --split::tools::split_compact_raw
 *    --split::tools::split_compact_raw
 *    --split::tools::split_prefix_scan_raw
 *    --split::tools::split_prefix_scan
 *    --split::tools::split_prescan
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 * */
#pragma once 
#include "split_allocators.h"
#define SPLIT_VOTING_MASK 0xFFFFFFFF //32-bit wide for cuda warps not sure what we do on AMD HW
#define NUM_BANKS 32 //TODO depends on device
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

namespace split{
   namespace tools{
    
      inline bool isPow2(const size_t val ) {
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
                      split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>* output,
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


      /*
        Prefix Scan routine with Bank Conflict Optimization
        Credits to  https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
      */
      template <typename T>
      __global__
      void split_prescan(T* input,T* output,T* partial_sums, int n,size_t len){

         extern __shared__ T buffer[];
         int tid = threadIdx.x;
         int offset=1;
         int local_start=blockIdx.x*n;
         input=input+local_start;
         output=output+local_start;


         //Load into shared memory 
         int ai = tid;
         int bi = tid + (n/2);
         int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
         int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
         
         if (local_start+ai<len && local_start+bi<len){
            buffer[ai + bankOffsetA] = input[ai];
            buffer[bi + bankOffsetB] = input[bi];
         }

         //Reduce Phase
         for (int d =n>>1; d>0; d>>=1){
            __syncthreads();

            if (tid<d){
               int ai = offset*(2*tid+1)-1;
               int bi = offset*(2*tid+2)-1;
               ai += CONFLICT_FREE_OFFSET(ai);
               bi += CONFLICT_FREE_OFFSET(bi);

               buffer[bi]+=buffer[ai];
            }
            offset*=2;
         }
         
         //Exclusive scan so zero out last element (will propagate to first)
         if (tid==0){ partial_sums[blockIdx.x]=buffer[n-1+CONFLICT_FREE_OFFSET(n-1)] ;buffer[n-1+CONFLICT_FREE_OFFSET(n-1)]=0;}

         //Downsweep Phase
         for (int d =1; d<n; d*=2){

            offset>>=1;
            __syncthreads();
            if (tid<d){
               int ai = offset*(2*tid+1)-1;
               int bi = offset*(2*tid+2)-1;
               ai += CONFLICT_FREE_OFFSET(ai);
               bi += CONFLICT_FREE_OFFSET(bi);

               T tmp=buffer[ai];
               buffer[ai]=buffer[bi];
               buffer[bi]+=tmp;
            }
         }
         __syncthreads();
         if (local_start+ai<len && local_start+bi<len){
            output[ai] = buffer[ai + bankOffsetA];
            output[bi] = buffer[bi + bankOffsetB];
         }
      }


      template <typename T,typename Rule,size_t BLOCKSIZE=1024, size_t WARP=32>
      __global__
      void split_compact(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                         split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>* counts,
                         split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>* offsets,
                         split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* output,
                         Rule rule)
      {
         extern __shared__ uint32_t buffer[];
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


      template <typename T,typename U,typename Rule,size_t BLOCKSIZE=1024, size_t WARP=32>
      __global__
      void split_compact_keys(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>* input,
                         split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>* counts,
                         split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>* offsets,
                         split::SplitVector<U,split::split_unified_allocator<U>,split::split_unified_allocator<size_t>>* output,
                         Rule rule)
      {
         extern __shared__ uint32_t buffer[];
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
            output->at(private_index) = (input->at(tid)).first;
         }

         __syncthreads();
         if (tid==0){
            const unsigned int actual_total_blocks=offsets->back()+counts->back();
            output->erase(&output->at(actual_total_blocks),output->end());
         }

      }


      template <typename T, size_t BLOCKSIZE=1024,size_t WARP=32>
      void split_prefix_scan(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                             split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& output,
                             cudaStream_t s=0)

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

         //If the elements are too few manually override and launch a small kernel
         if (input_size<scanElements){
            scanBlocksize=input_size/2;
            scanElements=input_size;
            if (scanBlocksize==0){scanBlocksize+=1;}
            gridSize=1;
         }

         //Allocate memory for partial sums
         vector partial_sums(gridSize); 

         //TODO +FIXME extra shmem allocations
         split::tools::split_prescan<<<gridSize,scanBlocksize,2*scanElements*sizeof(T),s>>>(input.data(),output.data(),partial_sums.data(),scanElements,input.size());
         cudaStreamSynchronize(s);

         if (gridSize>1){
            if (partial_sums.size()<=scanElements){
               vector partial_sums_dummy(gridSize); 
               //TODO +FIXME extra shmem allocations
               split::tools::split_prescan<<<1,scanBlocksize,2*scanElements*sizeof(T),s>>>(partial_sums.data(),partial_sums.data(),partial_sums_dummy.data(),gridSize,partial_sums.size());
               cudaStreamSynchronize(s);
            }else{
               vector partial_sums_clone(partial_sums);
               split_prefix_scan<T,BLOCKSIZE,WARP>(partial_sums_clone,partial_sums,s);
            }
            split::tools::scan_add<<<gridSize,scanBlocksize,0,s>>>(output.data(),partial_sums.data(),scanElements,output.size());
            cudaStreamSynchronize(s);
         }
      }

      
      template <typename T, typename Rule,size_t BLOCKSIZE=1024,size_t WARP=32>
      void copy_if(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                   split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& output,
                   Rule rule,
                   cudaStream_t s=0)
      {


         using vector=split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>;
         using vector_int=split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>;
         
         //Figure out Blocks to use
         size_t nBlocks=input.size()/BLOCKSIZE; 
         if (nBlocks==0){nBlocks+=1;}
         vector_int counts(nBlocks);
         vector_int offsets(nBlocks);

               
         //Phase 1 -- Calculate per warp workload
         vector * d_input=input.upload();
         vector_int * d_counts=counts.upload();
         split::tools::scan_reduce<<<nBlocks,BLOCKSIZE,0,s>>>(d_input,d_counts,rule);
         cudaStreamSynchronize(s);
         cudaFree(d_input);
         cudaFree(d_counts);


         //Step 2 -- Exclusive Prefix Scan on offsets
         if (nBlocks==1){
            split_prefix_scan<uint32_t,2,WARP>(counts,offsets,s);
         }else{
            split_prefix_scan<uint32_t,BLOCKSIZE,WARP>(counts,offsets,s);
         }
         cudaStreamSynchronize(s);


         //Step 3 -- Compaction
         vector* d_output=output.upload();
         vector_int* d_offsets=offsets.upload();
         d_input=input.upload();
         d_counts=counts.upload();
         split::tools::split_compact<T,Rule,BLOCKSIZE,WARP><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int),s>>>(d_input,d_counts,d_offsets,d_output,rule);
         cudaStreamSynchronize(s);
         //Deallocate the handle pointers

         cudaFree(d_input);
         cudaFree(d_counts);
         cudaFree(d_output);
         cudaFree(d_offsets);
      }

      template <typename T,typename U, typename Rule,size_t BLOCKSIZE=1024,size_t WARP=32>
      void copy_keys_if(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                   split::SplitVector<U,split::split_unified_allocator<U>,split::split_unified_allocator<size_t>>& output,
                   Rule rule,
                   cudaStream_t s=0)
      {


         using vector=split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>;
         using keyvector=split::SplitVector<U,split::split_unified_allocator<U>,split::split_unified_allocator<size_t>>;
         using vector_int=split::SplitVector<uint32_t,split::split_unified_allocator<uint32_t>,split::split_unified_allocator<size_t>>;
         
         //Figure out Blocks to use
         size_t nBlocks=input.size()/BLOCKSIZE; 
         if (nBlocks==0){nBlocks+=1;}
         vector_int counts(nBlocks);
         vector_int offsets(nBlocks);

               
         //Phase 1 -- Calculate per warp workload
         vector * d_input=input.upload();
         vector_int * d_counts=counts.upload();
         split::tools::scan_reduce<<<nBlocks,BLOCKSIZE,0,s>>>(d_input,d_counts,rule);
         cudaStreamSynchronize(s);
         cudaFree(d_input);
         cudaFree(d_counts);


         //Step 2 -- Exclusive Prefix Scan on offsets
         if (nBlocks==1){
            split_prefix_scan<uint32_t,2,WARP>(counts,offsets,s);
         }else{
            split_prefix_scan<uint32_t,BLOCKSIZE,WARP>(counts,offsets,s);
         }
         cudaStreamSynchronize(s);


         //Step 3 -- Compaction
         keyvector* d_output=output.upload();
         vector_int* d_offsets=offsets.upload();
         d_input=input.upload();
         d_counts=counts.upload();
         split::tools::split_compact_keys<T,U,Rule,BLOCKSIZE,WARP><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int),s>>>(d_input,d_counts,d_offsets,d_output,rule);
         cudaStreamSynchronize(s);
         //Deallocate the handle pointers

         cudaFree(d_input);
         cudaFree(d_counts);
         cudaFree(d_output);
         cudaFree(d_offsets);
      }

      
      template <typename T,typename Rule,size_t BLOCKSIZE=1024, size_t WARP=32>
      __global__
      void split_compact_raw(T* input, uint32_t* counts, uint32_t* offsets, T* output, Rule rule,const size_t size,size_t nBlocks,uint32_t* retval){

         extern __shared__ uint32_t buffer[];
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         if (tid>=size) {return;}
         unsigned int offset = BLOCKSIZE/WARP;
         const unsigned int wid = tid/WARP;
         const unsigned int widb = threadIdx.x/WARP;
         const unsigned int w_tid=tid%WARP;
         const unsigned int warps_in_block = blockDim.x/WARP;
         const bool tres=rule(input[tid]);

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
         const unsigned int private_index	= buffer[offset+widb] + offsets[(wid/warps_in_block)] + neighbor_count ;
         if (tres && widb!=warps_in_block){
            output[private_index] = input[tid];
         }
         if (tid==0){
            //const unsigned int actual_total_blocks=offsets->back()+counts->back();
            *retval=offsets[nBlocks-1]+counts[nBlocks-1];
         }
      }


      class Cuda_mempool{
         private:
            size_t total_bytes;
            size_t bytes_used;
            void* _data;
            cudaStream_t s;

         public:
            explicit Cuda_mempool(size_t bytes,cudaStream_t str){
               s=str;
               cudaMallocAsync(&_data, bytes,s);
               CheckErrors("Cuda Memory Allocation");
               total_bytes=bytes;
               bytes_used=0;
            }
            Cuda_mempool(const Cuda_mempool& other)=delete;
            Cuda_mempool(Cuda_mempool&& other)=delete;
            ~Cuda_mempool(){
               cudaFreeAsync(_data,s);
            }

            void* allocate(const size_t bytes){
               assert(bytes_used+bytes<total_bytes && "Mempool run out of space and crashed!");
               bytes_used+=bytes;
               return (char*)_data+bytes_used;
            };

            void deallocate(const size_t bytes){
               bytes_used-=bytes;
            };

            void reset(){
               bytes_used=0;
            }

            const size_t& fill()const{return bytes_used;}
            const size_t& capacity()const{return total_bytes;}
      };



      template<typename T,typename Rule>
      __global__
      void scan_reduce_raw(T* input, uint32_t* output, Rule rule,size_t size){

         size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         if (tid<size){
            size_t total_valid_elements=__syncthreads_count(rule(input[tid]));
            if (threadIdx.x==0){
               output[blockIdx.x]=total_valid_elements;
            }
         }
      }


      template <typename T, size_t BLOCKSIZE=1024,size_t WARP=32>
      void split_prefix_scan_raw(T* input, T* output ,Cuda_mempool& mPool, const size_t input_size,cudaStream_t s=0){

         //Scan is performed in half Blocksizes
         size_t scanBlocksize= BLOCKSIZE/2;
         size_t scanElements=2*scanBlocksize;
         size_t gridSize = input_size/scanElements;

         //If input is not exactly divisible by scanElements we launch an extra block
         assert(isPow2(input_size) && "Using prefix scan with non powers of 2 as input size is not thought out yet :D");
         if (input_size%scanElements!=0){
            gridSize=1<<((int)ceil(log(++gridSize)/log(2)));
         }
         //If the elements are too few manually override and launch a small kernel
         if (input_size<scanElements){
            scanBlocksize=input_size/2;
            scanElements=input_size;
            if (scanBlocksize==0){scanBlocksize+=1;}
            gridSize=1;
         }
         

         //Allocate memory for partial sums
         T* partial_sums = (T*)mPool.allocate(gridSize*sizeof(T)); 
         //TODO + FIXME extra shmem
         split::tools::split_prescan<<<gridSize,scanBlocksize,2*scanElements*sizeof(T),s>>>(input,output,partial_sums,scanElements,input_size);
         cudaStreamSynchronize(s);


         if (gridSize>1){
            if (gridSize<=scanElements){
               T* partial_sums_dummy=(T*)mPool.allocate(sizeof(T));
               //TODO + FIXME extra shmem
               split::tools::split_prescan<<<1,scanBlocksize,2*scanElements*sizeof(T),s>>>(partial_sums,partial_sums,partial_sums_dummy,gridSize,gridSize*sizeof(T));
               cudaStreamSynchronize(s);
            }else{
               T* partial_sums_clone=(T*)mPool.allocate(gridSize*sizeof(T));
               cudaMemcpy(partial_sums_clone, partial_sums, gridSize*sizeof(T),cudaMemcpyDeviceToDevice);
               split_prefix_scan_raw(partial_sums_clone,partial_sums,mPool,gridSize,s);
               
            }
            split::tools::scan_add<<<gridSize,scanBlocksize,0,s>>>(output,partial_sums,scanElements,input_size);
            cudaStreamSynchronize(s);
         }
      }

      template <typename T, typename Rule,size_t BLOCKSIZE=1024,size_t WARP=32>
      uint32_t copy_if_raw(split::SplitVector<T,split::split_unified_allocator<T>,split::split_unified_allocator<size_t>>& input,
                   T* output,
                   Rule rule,
                   cudaStream_t s=0)
      {
         
         //Figure out Blocks to use
         size_t nBlocks=input.size()/BLOCKSIZE; 
         if (nBlocks==0){nBlocks+=1;}
         
         //Allocate with Mempool
         const size_t memory_for_pool = 8*nBlocks*sizeof(uint32_t) ;
         Cuda_mempool mPool(memory_for_pool,s);

         uint32_t* d_counts;
         uint32_t* d_offsets;
         cudaStreamSynchronize(s);
         d_counts=(uint32_t*)mPool.allocate(nBlocks*sizeof(uint32_t));

               
         //Phase 1 -- Calculate per warp workload
         split::tools::scan_reduce_raw<<<nBlocks,BLOCKSIZE,0,s>>>(input.data(),d_counts,rule,input.size());
         d_offsets=(uint32_t*)mPool.allocate(nBlocks*sizeof(uint32_t));
         cudaStreamSynchronize(s);


         //Step 2 -- Exclusive Prefix Scan on offsets
         if (nBlocks==1){
            split_prefix_scan_raw<uint32_t,2,WARP>(d_counts,d_offsets,mPool,nBlocks,s);
         }else{
            split_prefix_scan_raw<uint32_t,BLOCKSIZE,WARP>(d_counts,d_offsets,mPool,nBlocks,s);
         }
         cudaStreamSynchronize(s);


         //Step 3 -- Compaction
         uint32_t* retval=(uint32_t*)mPool.allocate(sizeof(uint32_t));
         split::tools::split_compact_raw<T,Rule,BLOCKSIZE,WARP><<<nBlocks,BLOCKSIZE,2*(BLOCKSIZE/WARP)*sizeof(unsigned int),s>>>(input.data(),d_counts,d_offsets,output,rule,input.size(),nBlocks,retval);
         cudaStreamSynchronize(s);
         uint32_t numel;
         cudaMemcpy(&numel,retval,sizeof(uint32_t),cudaMemcpyDeviceToHost);
         cudaStreamSynchronize(s);
         return numel;
      }
   }//namespace tools
}//namespace split

/* File:    hashers.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 * Description: Defines parallel hashers that insert,retrieve and 
 *               delete elements to/from Hahsinator on device
 *
 * This file defines the following:
 *    --Hashinator::Hashers::reset_to_empty()
 *    --Hashinator::Hashers::insert
 *    --Hashinator::Hashers::insert_kernel()
 *    --Hashinator::Hashers::retrieve_kernel()
 *    --Hashinator::Hashers::delete_kernel()
 *    --Hashinator::Hashers::insert()
 *    --Hashinator::Hashers::retrieve()
 *    --Hashinator::Hashers::erase()
 *    --Hashinator::Hashers::launchParams()
 *    --Hashinator::Hashers::warpReduce()
 *    --Hashinator::Hashers::warpReduceMax()
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
#include "hashfunctions.h"
#include "defaults.h"
#include "../common.h"
#include "../hashinator_atomics.h"

namespace Hashinator{

   namespace Hashers{

      /*
       * Resets all elements pointed by src to EMPTY in dst 
       * If an elements in src is not found this will assert(false)
       * */
      template<typename KEY_TYPE, 
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>>
      __global__ 
      void reset_to_empty(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                          hash_pair<KEY_TYPE, VAL_TYPE>* dst,
                          const int sizePower,
                          size_t maxoverflow,
                          size_t Nsrc)

      {
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         if (tid>=Nsrc){return ;}
         hash_pair<KEY_TYPE,VAL_TYPE>&candidate=src[tid];
         int bitMask = (1 <<(sizePower )) - 1; 
         auto hashIndex = HashFunction::_hash(candidate.first,sizePower);

         for(size_t i =0; i< (1<<sizePower);++i){
            uint32_t probing_index=(hashIndex+i)&bitMask;
            KEY_TYPE old = h_atomicCAS(&dst[probing_index].first,candidate.first,EMPTYBUCKET);
            if (old==candidate.first){
               return ;
            }
         }
         assert(false && "Could not reset element. Something is broken!");
      }

      /*
       * Resets all elements in dst to EMPTY, VAL_TYPE()
       * */
      template<typename KEY_TYPE, typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max()>
      __global__ 
      void reset_all_to_empty(hash_pair<KEY_TYPE, VAL_TYPE>* dst,
                              const size_t len, size_t * fill)
      {
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         //Early exit here
         if (tid>=len){return;}
         if(dst[tid].first!=EMPTYBUCKET){
            dst[tid].first=EMPTYBUCKET;
            dst[tid].second=VAL_TYPE();
            h_atomicSub((unsigned int*)fill,1);
         }
         return;
      }

      template< int WARPSIZE>
      HASHINATOR_DEVICEONLY
      __forceinline__
      int warpReduce(int localCount){
         for (int i=WARPSIZE/2; i>0; i=i/2){
            localCount += h_shuffle_down(localCount, i,SPLIT_VOTING_MASK);
         }
         return localCount;
      }

      template< int WARPSIZE>
      HASHINATOR_DEVICEONLY
      __forceinline__
      uint64_t warpReduceMax(uint64_t entry){
         for (int i=WARPSIZE/2; i>0; i=i/2){
            entry= std::max( entry ,  h_shuffle_down(entry, i,SPLIT_VOTING_MASK));
         }
         return entry;
      }

#ifdef __NVCC__
      template<typename KEY_TYPE, 
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                         hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                         int sizePower,
                         size_t maxoverflow,
                         size_t* d_overflow,
                         size_t* d_fill,
                         size_t len,status* err)
      {

         __shared__ uint32_t addMask[WARPSIZE];
         __shared__ uint64_t warpOverflow[WARPSIZE];

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         const size_t proper_w_tid=tid%WARPSIZE; //the proper WID as if we had no Virtual warps
         const size_t proper_wid=tid/WARPSIZE; 
         const size_t blockWid=proper_wid%WARPSIZE;
 
            
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }
         
         //Zero out shared count;
         if (proper_w_tid==0 && blockWid==0 ){
            for (int i =0 ; i< WARPSIZE; i++){
               addMask[i] = 0;
               warpOverflow[i] = 0;
            }
         }
         __syncthreads();

         

         uint32_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }

         hash_pair<KEY_TYPE,VAL_TYPE> candidate=src[wid];
         const int  bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidate.first,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;
         uint32_t vWarpDone=0;  // state of virtual warp
         uint32_t localCount=0;
         uint64_t threadOverflow=1;

         for(size_t i=0; i<(1<<sizePower); i+=VIRTUALWARP){

            //Check if this virtual warp is done. 
            if (vWarpDone){
               break;
            }

            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            auto target=buckets[probingindex];

            //vote for available emptybuckets in warp region
            //Note that this has to be done before voting for already existing elements (below)
            auto mask = warpVote(target.first==EMPTYBUCKET,submask);

            //Check if this elements already exists
            auto already_exists = warpVote(target.first==candidate.first,submask);
            if (already_exists){
               int winner =findFirstSig( already_exists ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  h_atomicExch(&buckets[probingindex].second,candidate.second);
                  //This virtual warp is now done.
                  vWarpDone = 1; 
               }
            }

            //If any duplicate was there now is the time for the whole Virtual warp to find out!
            vWarpDone=warpVoteAny(vWarpDone,submask);

            while(mask && !vWarpDone){
               int winner =findFirstSig( mask ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = h_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     threadOverflow = (probingindex<optimalindex)?(1<<sizePower):(probingindex-optimalindex);
                     h_atomicExch(&buckets[probingindex].second,candidate.second);
                     vWarpDone=1;
                     //Flip the bit which corresponds to the thread that added an element
                     localCount++;
                  }else if (old==candidate.first){
                     //Parallel stuff are fun. Major edge case!
                     h_atomicExch(&buckets[probingindex].second,candidate.second);
                     vWarpDone=1;
                  }
               }
               //If any of the virtual warp threads are done the the whole 
               //Virtual warp is done
               vWarpDone=warpVoteAny(vWarpDone,submask);
               mask ^= (1UL << winner);
            }
         }

        
         /*
            Update fill and overflow in 2 steps:
            Step 1--> First thread per warp reduces the total elements added (per Warp)
            Step 2--> Reduce the blockTotal from the warpTotals but do it in registers using the first warp in the block
         */
         
         //Per warp reduction
         __syncwarp();
         int warpTotals= warpReduce<WARPSIZE>(localCount);
         uint64_t perWarpOverflow=warpReduceMax<WARPSIZE>(threadOverflow);
         __syncwarp();

         //Store to shmem minding Bank Conflicts
         if (proper_w_tid==0){
            //Write the count to the same place 
            addMask[(blockWid)]=warpTotals;
            warpOverflow[( blockWid )]=perWarpOverflow;         
         }

         __syncthreads();
         //First warp in block reductions
         if (blockWid==0){
            uint64_t blockOverflow=warpReduceMax<WARPSIZE>(warpOverflow[( proper_w_tid )]);
            int blockTotal = warpReduce<WARPSIZE>(addMask[(proper_w_tid)]);
            //First thread updates fill and overlfow (1 update per block)
            if (proper_w_tid==0){
               atomicMax(( unsigned long long*)d_overflow,(unsigned long long)nextPow2(blockOverflow));
               h_atomicAdd(d_fill,blockTotal);;
            }
         }

         //Make sure everyone actually made it otherwise raise the error flag.
         if (warpVote(vWarpDone,SPLIT_VOTING_MASK)!=__activemask()){
            h_atomicExch((uint32_t*)err,(uint32_t)status::fail);
         }
         return;
      }
      

      /*Warp Synchronous hashing kernel for hashinator's internal use:
       * This method uses 32-thread Warps to hash an element from src.
       * Threads in a given warp simultaneously try to hash an element
       * in the buckets by using warp voting to communicate available 
       * positions in the probing  sequence. The position of least overflow
       * is selected by using __ffs to decide on the winner. If no positios
       * are available in the probing sequence the warp shifts by a warp size
       * and ties to overflow(up to maxoverflow).
       * No tombstones allowed!
       * Parameters:
       *    src          -> pointer to device data with pairs to be inserted
       *    buckets      -> current hashinator buckets
       *    sizePower    -> current hashinator sizepower
       *    maxoverflow  -> maximum allowed overflow
       *    d_overflow   -> stores the overflow after inserting the elements
       *    d_fill       -> stores the device fill after inserting the elements
       *    len          -> number of elements to read from src
       * */
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(KEY_TYPE* keys,
                         VAL_TYPE* vals,
                         hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                         int sizePower,
                         size_t maxoverflow,
                         size_t* d_overflow,
                         size_t* d_fill,
                         size_t len,status* err)
      {
         
         __shared__ uint32_t addMask[WARPSIZE];
         __shared__ uint64_t warpOverflow[WARPSIZE];

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         const size_t proper_w_tid=tid%WARPSIZE; //the proper WID as if we had no Virtual warps
         const size_t proper_wid=tid/WARPSIZE; 
         const size_t blockWid=proper_wid%WARPSIZE;
 
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }
         
            
         //Zero out shared count;
         if (proper_w_tid==0 && blockWid==0 ){
            for (int i =0 ; i< WARPSIZE; i++){
               addMask[i] = 0;
               warpOverflow[i] = 0;
            }
         }
         __syncthreads();

         uint32_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }

         KEY_TYPE candidateKey=keys[wid];
         VAL_TYPE candidateVal=vals[wid];
         const int  bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidateKey,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;
         uint32_t vWarpDone=0;  // state of virtual warp
         uint32_t localCount=0;
         uint64_t threadOverflow=1;

         for(size_t i=0; i<(1<<sizePower); i+=VIRTUALWARP){

            //Check if this virtual warp is done. 
            if (vWarpDone){
               break;
            }
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            auto target=buckets[probingindex];

            //vote for available emptybuckets in warp region
            //Note that this has to be done before voting for already existing elements (below)
            auto mask = warpVote(target.first==EMPTYBUCKET,submask);

            //Check if this elements already exists
            auto already_exists =warpVote(target.first==candidateKey,submask);
            if (already_exists){
               int winner =findFirstSig( already_exists ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  h_atomicExch(&target.second,candidateVal);
                  //This virtual warp is now done.
                  vWarpDone = 1; 
               }
            }

            //If any duplicate was there now is the time for the whole Virtual warp to find out!
            vWarpDone=warpVoteAny(vWarpDone,submask);

            while(mask && !vWarpDone){
               int winner =findFirstSig( mask ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = h_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidateKey);
                  if (old == EMPTYBUCKET){
                     threadOverflow = (probingindex<optimalindex)?(1<<sizePower):(probingindex-optimalindex);
                     h_atomicExch(&buckets[probingindex].second,candidateVal);
                     vWarpDone=1;
                     //Flip the bit which corresponds to the thread that added an element
                     localCount++;
                  }else if (old==candidateKey){
                     //Parallel stuff are fun. Major edge case!
                     h_atomicExch(&buckets[probingindex].second,candidateVal);
                     vWarpDone=1;
                  }
               }
               //If any of the virtual warp threads are done the the whole 
               //Virtual warp is done
               vWarpDone=warpVoteAny(vWarpDone,submask);
               mask ^= (1UL << winner);
            }
         }

         /*
            Update fill and overflow in 2 steps:
            Step 1--> First thread per warp reduces the total elements added (per Warp)
            Step 2--> Reduce the blockTotal from the warpTotals but do it in registers using the first warp in the block
         */
         __syncwarp();
         //Per warp reduction
         int warpTotals= warpReduce<WARPSIZE>(localCount);
         uint64_t perWarpOverflow=warpReduceMax<WARPSIZE>(threadOverflow);

         //Store to shmem minding Bank Conflicts
         if (proper_w_tid==0){
            //Write the count to the same place 
            addMask[(blockWid)]=warpTotals;
            warpOverflow[( blockWid )]=perWarpOverflow;         
         }

         __syncthreads();
         //First warp in block reductions
         if (blockWid==0){
            uint64_t blockOverflow=warpReduceMax<WARPSIZE>(warpOverflow[( proper_w_tid )]);
            int blockTotal = warpReduce<WARPSIZE>(addMask[(proper_w_tid)]);
            //First thread updates fill and overlfow (1 update per block)
            if (proper_w_tid==0){
               atomicMax(( unsigned long long*)d_overflow,(unsigned long long)nextPow2(blockOverflow));
               h_atomicAdd(d_fill,blockTotal);;
            }
         }

         //Make sure everyone actually made it otherwise raise the error flag.
         if (warpVote(vWarpDone,SPLIT_VOTING_MASK)!=__activemask()){
            h_atomicExch((uint32_t*)err,(uint32_t)status::fail);
         }
         return;

      }

#endif
#ifdef __HIP_PLATFORM_HCC___
      template<typename KEY_TYPE, 
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                         hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                         int sizePower,
                         size_t maxoverflow,
                         size_t* d_overflow,
                         size_t* d_fill,
                         size_t len,status* err)
      {


         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         const size_t proper_w_tid=tid%WARPSIZE; //the proper WID as if we had no Virtual warps
         const size_t proper_wid=tid/WARPSIZE; 
         const size_t blockWid=proper_wid%WARPSIZE;
 
            
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }
         
         

         uint64_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint64_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_AMD(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }

         hash_pair<KEY_TYPE,VAL_TYPE> candidate=src[wid];
         const int  bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidate.first,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;
         uint32_t localCount=0;
         uint64_t threadOverflow=1;
         uint64_t vWarpDone=0;  // state of virtual warp

         for(size_t i=0; i<(1<<sizePower); i+=VIRTUALWARP){

            //Check if this virtual warp is done. 
            if (vWarpDone){
               break;
            }

            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            auto target=buckets[probingindex];

            //vote for available emptybuckets in warp region
            //Note that this has to be done before voting for already existing elements (below)
            auto mask = warpVote(target.first==EMPTYBUCKET,submask)&submask;

            //Check if this elements already exists
            auto already_exists = warpVote(target.first==candidate.first,submask)&submask;
            if (already_exists){
               int winner =findFirstSig( already_exists ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  h_atomicExch(&buckets[probingindex].second,candidate.second);
                  //This virtual warp is now done.
                  vWarpDone = 1; 
               }
            }

            //If any duplicate was there now is the time for the whole Virtual warp to find out!
            vWarpDone=warpVote(vWarpDone>0,submask)&submask;

            while(mask && !vWarpDone){
               int winner =findFirstSig( mask ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = h_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     threadOverflow = (probingindex<optimalindex)?(1<<sizePower):(probingindex-optimalindex);
                     h_atomicExch(&buckets[probingindex].second,candidate.second);
                     vWarpDone=1;
                     //Flip the bit which corresponds to the thread that added an element
                     localCount++;
                  }else if (old==candidate.first){
                     //Parallel stuff are fun. Major edge case!
                     h_atomicExch(&buckets[probingindex].second,candidate.second);
                     vWarpDone=1;
                  }
               }
               //If any of the virtual warp threads are done the the whole 
               //Virtual warp is done
               vWarpDone=warpVote(vWarpDone>0,submask)&submask;
               mask ^= (1UL << winner);
            }
         }

        
         /*
            Update fill and overflow
            First thread per warp reduces the total elements added (per Warp)
         */
         //Per warp reduction
         int warpTotals= warpReduce<WARPSIZE>(localCount);
         uint64_t perWarpOverflow=warpReduceMax<WARPSIZE>(threadOverflow);

         //Update fill and overlfow
         if (proper_w_tid==0){
            h_atomicAdd(d_fill,warpTotals);;
            if (perWarpOverflow>*d_overflow){
               h_atomicExch(d_overflow,nextPow2(perWarpOverflow));
            }
         }
         return;
      }
      

      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(KEY_TYPE* keys,
                         VAL_TYPE* vals,
                         hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                         int sizePower,
                         size_t maxoverflow,
                         size_t* d_overflow,
                         size_t* d_fill,
                         size_t len,status* err)
      {
         

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         const size_t proper_w_tid=tid%WARPSIZE; //the proper WID as if we had no Virtual warps
         const size_t proper_wid=tid/WARPSIZE; 
         const size_t blockWid=proper_wid%WARPSIZE;
 
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }
         
         uint64_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint64_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_AMD(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }

         KEY_TYPE candidateKey=keys[wid];
         VAL_TYPE candidateVal=vals[wid];
         const int  bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidateKey,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;
         uint32_t localCount=0;
         uint64_t vWarpDone=0;  // state of virtual warp
         uint64_t threadOverflow=1;

         for(size_t i=0; i<(1<<sizePower); i+=VIRTUALWARP){

            //Check if this virtual warp is done. 
            if (vWarpDone){
               break;
            }
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            auto target=buckets[probingindex];

            //vote for available emptybuckets in warp region
            //Note that this has to be done before voting for already existing elements (below)
            auto mask = warpVote(target.first==EMPTYBUCKET,submask)&submask;

            //Check if this elements already exists
            auto already_exists =warpVote(target.first==candidateKey,submask)&submask;
            if (already_exists){
               int winner =findFirstSig( already_exists ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  h_atomicExch(&target.second,candidateVal);
                  //This virtual warp is now done.
                  vWarpDone = 1; 
               }
            }

            //If any duplicate was there now is the time for the whole Virtual warp to find out!
            vWarpDone=warpVote(vWarpDone>0,submask)&submask;

            while(mask && !vWarpDone){
               int winner =findFirstSig( mask ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = h_atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidateKey);
                  if (old == EMPTYBUCKET){
                     threadOverflow = (probingindex<optimalindex)?(1<<sizePower):(probingindex-optimalindex);
                     h_atomicExch(&buckets[probingindex].second,candidateVal);
                     vWarpDone=1;
                     //Flip the bit which corresponds to the thread that added an element
                     localCount++;
                  }else if (old==candidateKey){
                     //Parallel stuff are fun. Major edge case!
                     h_atomicExch(&buckets[probingindex].second,candidateVal);
                     vWarpDone=1;
                  }
               }
               //If any of the virtual warp threads are done the the whole 
               //Virtual warp is done
               vWarpDone=warpVote(vWarpDone>0,submask)&submask;
               mask ^= (1UL << winner);
            }
         }

         /*
            Update fill and overflow 
            First thread per warp reduces the total elements added (per Warp)
         */
         //Per warp reduction
         int warpTotals= warpReduce<WARPSIZE>(localCount);
         uint64_t perWarpOverflow=warpReduceMax<WARPSIZE>(threadOverflow);

         //Update fill and overlfow
         if (proper_w_tid==0){
            h_atomicAdd(d_fill,warpTotals);;
            if (perWarpOverflow>*d_overflow){
               h_atomicExch(d_overflow,nextPow2(perWarpOverflow));
            }
         }
         return;
      }

#endif
      /*
       * Similarly to the insert_kernel we examine elements in keys and return their value in vals,
       * if the do exist in the hashmap. If the elements is not found and invalid key is returned;
       * */
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void retrieve_kernel(KEY_TYPE* keys,
                           VAL_TYPE* vals,
                           hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                           int sizePower,
                           size_t maxoverflow)
      {

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         #ifdef __NVCC__
         uint32_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         #ifdef __HIP_PLATFORM_HCC___
         uint64_t     subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint64_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         KEY_TYPE& candidateKey=keys[wid];
         VAL_TYPE& candidateVal=vals[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidateKey,sizePower);

         //Check for duplicates
         for(size_t i=0; i<maxoverflow; i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            const auto  maskExists = warpVote(buckets[probingindex].first==candidateKey,SPLIT_VOTING_MASK)&submask;
            const auto  emptyFound = warpVote(buckets[probingindex].first==EMPTYBUCKET,SPLIT_VOTING_MASK)&submask;
            //If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
            if (!maskExists && emptyFound){
               return;
            }
            if (maskExists){
               int winner =findFirstSig( maskExists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  h_atomicExch(&candidateVal,buckets[probingindex].second);
               }
               return;
             }
         }

      }

      /*
       * Similarly to the insert_kernel we examine elements in keys and return their value in vals,
       * if the do exist in the hashmap. If the elements is not found and invalid key is returned;
       * */
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void retrieve_kernel(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                           hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                           int sizePower,
                           size_t maxoverflow)
      {

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         #ifdef __NVCC__
         uint32_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         #ifdef __HIP_PLATFORM_HCC___
         uint64_t     subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint64_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         hash_pair<KEY_TYPE,VAL_TYPE>&candidate=src[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const auto hashIndex = HashFunction::_hash(candidate.first,sizePower);

         //Check for duplicates
         for(size_t i=0; i<maxoverflow; i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            const auto  maskExists = warpVote(buckets[probingindex].first==candidate.first,SPLIT_VOTING_MASK)&submask;
            const auto  emptyFound = warpVote(buckets[probingindex].first==EMPTYBUCKET,SPLIT_VOTING_MASK)&submask;
            //If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
            if (!maskExists && emptyFound){
               return;
            }
            if (maskExists){
               int winner =findFirstSig( maskExists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  h_atomicExch(&candidate.second,buckets[probingindex].second);
               }
               return;
             }
         }

      }

      /*
       * In a similar way to the insert and retrieve kernels we 
       * delete keys in "keys" if they do exist in the hasmap.
       * If the keys do not exist we do nothing.
       * */
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               KEY_TYPE TOMBSTONE=EMPTYBUCKET-1,
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>,
               int WARPSIZE=defaults::WARPSIZE,
               int elementsPerWarp>
      __global__ 
      void delete_kernel(KEY_TYPE* keys,
                           hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                           size_t* d_tombstoneCounter,
                           int sizePower,
                           size_t maxoverflow,
                           size_t len)
      {

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         
         //Early quit if we have more warps than elements to handle
         if (wid>=len){
            return;
         }

         #ifdef __NVCC__
         uint32_t subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         #ifdef __HIP_PLATFORM_HCC___
         uint64_t     subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         uint64_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask_CUDA(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         #endif 
         

         KEY_TYPE& candidateKey=keys[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const auto  hashIndex = HashFunction::_hash(candidateKey,sizePower);

         for(size_t i=0; i<maxoverflow; i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            const auto  maskExists = warpVote(buckets[probingindex].first==candidateKey,SPLIT_VOTING_MASK)&submask;
            const auto  emptyFound = warpVote(buckets[probingindex].first==EMPTYBUCKET,SPLIT_VOTING_MASK)&submask;
            //If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
            if (!maskExists && emptyFound){
               return;
            }
            if (maskExists){
               int winner =findFirstSig( maskExists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  h_atomicExch(&buckets[probingindex].first, TOMBSTONE);
                  h_atomicAdd(d_tombstoneCounter, 1);
               }
               return;
             }
         }
      }



      template<typename KEY_TYPE,
               typename VAL_TYPE,
               class HashFunction,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               KEY_TYPE TOMBSTONE=EMPTYBUCKET=1,
               int WARP=defaults::WARPSIZE,
               int elementsPerWarp=1>
      class Hasher{
      
      //Make sure we have sane elements per warp
      static_assert(elementsPerWarp>0 && elementsPerWarp<=WARP && "Device hasher cannot be instantiated");

      public:

         //Overload with separate input for keys and values.
         static void insert(KEY_TYPE* keys,
                            VAL_TYPE* vals,
                            hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                            int sizePower,
                            size_t maxoverflow,
                            size_t* d_overflow,
                            size_t* d_fill,
                            size_t len,
                            status* err,
                            cudaStream_t s=0)
         {
            size_t blocks,blockSize;
            *err=status::success;
            launchParams(len,blocks,blockSize);
            insert_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,vals,buckets,sizePower,maxoverflow,d_overflow,d_fill,len,err);
            cudaStreamSynchronize(s);
            #ifndef NDEBUG
            if (*err==status::fail){
               std::cerr<<"***** Hashinator Runtime Warning ********"<<std::endl;
               std::cerr<<"Warning: Hashmap completely overflown in Device Insert.\nNot all ellements were inserted!\nConsider resizing before calling insert"<<std::endl;
               std::cerr<<"******************************"<<std::endl;
            }
            #endif
         }
         
         //Overload with hash_pair<key,val> (k,v) inputs
         //Used by the tombstone cleaning method.
         static void insert(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                            hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                            int sizePower,
                            size_t maxoverflow,
                            size_t* d_overflow,
                            size_t* d_fill,
                            size_t len,
                            status* err,
                            cudaStream_t s=0)
         {
            size_t blocks,blockSize;
            *err=status::success;
            launchParams(len,blocks,blockSize);
            insert_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (src,buckets,sizePower,maxoverflow,d_overflow,d_fill,len,err);
            cudaStreamSynchronize(s);
            #ifndef NDEBUG
            if (*err==status::fail){
               std::cerr<<"***** Hashinator Runtime Warning ********"<<std::endl;
               std::cerr<<"Warning: Hashmap completely overflown in Device Insert.\nNot all ellements were inserted!\nConsider resizing before calling insert"<<std::endl;
               std::cerr<<"******************************"<<std::endl;
            }
            #endif
         }

         //Retrieve wrapper
         static void retrieve(KEY_TYPE* keys,
                              VAL_TYPE* vals,
                              hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                              int sizePower,
                              size_t maxoverflow,
                              size_t len,
                              cudaStream_t s=0)
         {

            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            retrieve_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,vals,buckets,sizePower,maxoverflow);
            cudaStreamSynchronize(s);

         }

         static void retrieve(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                              hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                              int sizePower,
                              size_t maxoverflow,
                              size_t len,
                              cudaStream_t s=0)
         {

            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            retrieve_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (src,buckets,sizePower,maxoverflow);
            cudaStreamSynchronize(s);

         }
         //Delete wrapper
         static void erase(KEY_TYPE* keys,
                           hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                           size_t* d_tombstoneCounter,
                           int sizePower,
                           size_t maxoverflow,
                           size_t len,
                           cudaStream_t s=0)
         {

            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            delete_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,TOMBSTONE,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,buckets,d_tombstoneCounter,sizePower,maxoverflow,len);
            cudaStreamSynchronize(s);

         }

      private:
         static void launchParams(size_t N,size_t& blocks,size_t& blockSize){
            //fast ceil for positive ints
            size_t warpsNeeded=N/elementsPerWarp + (N%elementsPerWarp!=0);
            blockSize=std::min(warpsNeeded*WARP,(size_t)Hashinator::defaults::MAX_BLOCKSIZE);
            blocks=warpsNeeded*WARP/blockSize + ((warpsNeeded*WARP)%blockSize!=0);
            //blocks*=2;
            return;
         }
      };

   } //namespace Hashers
}//namespace Hashinator

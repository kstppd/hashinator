/* File:    hashers.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines parallel hashers that insert elements
 *              to Hahsinator on device
 *
 * This file defines the following:
 *    --Hashinator::Hashers::reset_to_empty()
 *    --Hashinator::Hashers::hasher_V2()
 *
 * (c) Copyright 2012-2023 Apache License version 2.0 or later
 * */
#pragma once
#include "hashfunctions.h"

namespace Hashinator{

   namespace Hashers{

      template<typename KEY_TYPE, 
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Murmur<KEY_TYPE>>
      __global__ 
      void reset_to_empty(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                          hash_pair<KEY_TYPE, VAL_TYPE>* dst,
                          const int sizePower,
                          int maxoverflow,
                          size_t Nsrc)

      {
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         if (tid>=Nsrc){return ;}
         hash_pair<KEY_TYPE,VAL_TYPE>&candidate=src[tid];
         int bitMask = (1 <<(sizePower )) - 1; 
         uint32_t hashIndex = HashFunction::_hash(candidate.first);
         uint32_t actual_index=(hashIndex+candidate.offset)&bitMask;
         atomicCAS(&dst[actual_index].first,candidate.first,EMPTYBUCKET);
         return ;
      }


      /* Warp Synchronous hashing kernel for hashinator's internal use:
       * This method uses 32-thread Warps to hash an element from src.
       * Threads in a given warp simultaneously try to hash an element
       * in the buckets by using warp voting to communicate available 
       * positions in the probing  sequence. The position of least overflow
       * is selected by using __ffs to decide on the winner. If no positions
       * are available in the probing sequence the warp shifts by a warp size
       * and tries to overflow(up to maxoverflow).
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
      template<typename KEY_TYPE, typename VAL_TYPE,KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),class HashFunction=HashFunctions::Murmur<KEY_TYPE>,int WARP=32>
      __global__ 
      void hasher_V2(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                    hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                    int sizePower,
                    int maxoverflow,
                    int* d_overflow,
                    size_t* d_fill,
                    size_t len)
      {

         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/WARP;
         const size_t w_tid=tid%WARP;
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }
                               
         hash_pair<KEY_TYPE,VAL_TYPE>&candidate=src[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidate.first);
         const size_t optimalindex=(hashIndex) & bitMask;
         bool done=false;


         for(int i=0; i<(1<<sizePower); i+=WARP){

            //Get the position we should be looking into
            size_t vecindex=((hashIndex+i+w_tid) & bitMask ) ;
            
            //vote for already existing in warp region
            uint32_t mask_already_exists = __ballot_sync(SPLIT_VOTING_MASK,buckets[vecindex].first==candidate.first);
            if (mask_already_exists){
               int winner =__ffs ( mask_already_exists ) -1;
               if(w_tid==winner){
                  atomicExch(&buckets[vecindex].second,candidate.second);
                  done=true;
               }
               int warp_done=__any_sync(__activemask(),done);
               if(warp_done>0){
                  return;
               }
            }

            //vote for available emptybuckets in warp region
            uint32_t mask = 1;//_ballot_sync(SPLIT_VOTING_MASK,buckets[vecindex].first==EMPTYBUCKET);
            while(mask!=0){
               mask = __ballot_sync(SPLIT_VOTING_MASK,buckets[vecindex].first==EMPTYBUCKET);
               int winner =__ffs ( mask ) -1;
               if (w_tid==winner){
                  KEY_TYPE old = atomicCAS(&buckets[vecindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     //TODO the atomicExch here could be non atomics as no other thread can probe here
                     int overflow = vecindex-optimalindex;
                     atomicExch(&buckets[vecindex].second,candidate.second);
                     atomicExch(&buckets[vecindex].offset,overflow);
                     atomicMax((int*)d_overflow,overflow);
                     atomicAdd((unsigned long long int*)d_fill, 1);
                     done=true;
                  }
               }
               int warp_done=__any_sync(__activemask(),done);
               if(warp_done>0){
                  return;
               }
            }
         }
         return ;
      }


      template<typename KEY_TYPE,
               typename VAL_TYPE,
               class HashFunction,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               int WARP=32>
      class Hasher{

      public:
         static void insert(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                            hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                            int sizePower,
                            int maxoverflow,
                            int* d_overflow,
                            size_t* d_fill,
                            size_t len)
         {
            size_t blocks,blockSize;
            launchParams(1,len,blocks,blockSize);
            hasher_V2<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,32><<<blocks,blockSize>>>(src,buckets,sizePower,maxoverflow,d_overflow,d_fill,len);
            cudaDeviceSynchronize();
         }

      private:
         static void launchParams(size_t elementsPerWarp,size_t N,size_t& blocks,size_t& blockSize){
            //fast ceil for positive ints
            size_t warpsNeeded=N/elementsPerWarp + (N%elementsPerWarp!=0);
            blockSize=std::min(warpsNeeded*WARP,(size_t)1024);
            blocks=warpsNeeded*WARP/blockSize + ((warpsNeeded*WARP)%blockSize!=0);
            blocks*=2;
            return;
         }
      };

   } //namespace Hashers
}//namespace Hashinator

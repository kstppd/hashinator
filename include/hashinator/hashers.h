/* File:    hashers.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines parallel hashers that insert elements
 *              to Hahsinator on device
 *
 * This file defines the following:
 *    --Hashinator::Hashers::reset_to_empty()
 *    --Hashinator::Hashers::hasher_V2()
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

namespace Hashinator{

   namespace Hashers{

      template<typename KEY_TYPE, 
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Fibonacci<KEY_TYPE>>
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
         uint32_t hashIndex = HashFunction::_hash(candidate.first,sizePower);

         for(size_t i =0; i< (1<<sizePower);++i){
            uint32_t probing_index=(hashIndex+i)&bitMask;
            KEY_TYPE old = atomicCAS(&dst[probing_index].first,candidate.first,EMPTYBUCKET);
            if (old==candidate.first){
               return ;
            }
         }
         assert(false && "Could not reset element. Something is broken!");
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
      template<typename KEY_TYPE, typename VAL_TYPE,KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),class HashFunction=HashFunctions::Murmur<KEY_TYPE>,int WARP=Hashinator::defaults::WARPSIZE>
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
         const size_t hashIndex = HashFunction::_hash(candidate.first,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;


         //Check for duplicates
         for(int i=0; i<(*d_overflow); i+=WARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;

            //vote for already existing in warp region
            uint32_t mask_already_exists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidate.first);
            if (mask_already_exists){
               int winner =__ffs ( mask_already_exists ) -1;
               if(w_tid==winner){
                  atomicExch(&buckets[probingindex].second,candidate.second);
               }
               return;
            }
         }

         //No duplicates so we insert
         bool done=false;
         for(int i=0; i<(1<<sizePower); i+=WARP){

            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            //vote for available emptybuckets in warp region
            uint32_t mask = 1;//_ballot_sync(SPLIT_VOTING_MASK,buckets[vecindex].first==EMPTYBUCKET);
            while(mask!=0){
               mask = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET);
               int winner =__ffs ( mask ) -1;
               if (w_tid==winner){
                  KEY_TYPE old = atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     //TODO the atomicExch here could be non atomics as no other thread can probe here
                     int overflow = probingindex-optimalindex;
                     atomicExch(&buckets[probingindex].second,candidate.second);
                     atomicMax((int*)d_overflow,overflow);
                     atomicAdd((unsigned long long int*)d_fill, 1);
                     done=true;
                  }
                  //Parallel stuff are fun. Major edge case!
                  if (old==candidate.first){
                     atomicExch(&buckets[probingindex].second,candidate.second);
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
      template<typename KEY_TYPE, typename VAL_TYPE,KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),class HashFunction=HashFunctions::Murmur<KEY_TYPE>,int WARPSIZE=32,int elementsPerWarp>
      __global__ 
      void hasher_V3(hash_pair<KEY_TYPE, VAL_TYPE>* src,
                    hash_pair<KEY_TYPE, VAL_TYPE>* buckets,
                    int sizePower,
                    int maxoverflow,
                    int* d_overflow,
                    size_t* d_fill,
                    size_t len)
      {
         
         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         unsigned int subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         
         //Early quit if we have more warps than elements to insert
         if (wid>=len){
            return;
         }

         auto getIntraWarpMask = [](unsigned int n ,unsigned int l ,unsigned int r)->unsigned int{
            int num = ((1<<r)-1)^((1<<(l-1))-1);
            return (n^num);
         };

         uint32_t submask=getIntraWarpMask(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         hash_pair<KEY_TYPE,VAL_TYPE>&candidate=src[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidate.first,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;

         //Check for duplicates
         for(int i=0; i<(*d_overflow); i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            uint32_t mask_already_exists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidate.first);
            mask_already_exists&=submask;

            if (mask_already_exists){
               int winner =__ffs ( mask_already_exists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  atomicExch(&buckets[probingindex].second,candidate.second);
               }
               return;
             }
         }


         //No duplicates so we insert
         bool done=false;
         for(int i=0; i<(1<<sizePower); i+=VIRTUALWARP){

            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            //vote for available emptybuckets in warp region
            uint32_t  mask;
            mask=1;
            while(mask!=0){
               mask = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET);
               mask&=submask;
               int winner =__ffs ( mask ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==winner){
                  KEY_TYPE old = atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     int overflow = probingindex-optimalindex;
                     atomicExch(&buckets[probingindex].second,candidate.second);
                     atomicMax((int*)d_overflow,overflow);
                     atomicAdd((unsigned long long int*)d_fill, 1);
                     done=true;
                  }
                  //Parallel stuff are fun. Major edge case!
                  if (old==candidate.first){
                     atomicExch(&buckets[probingindex].second,candidate.second);
                     done=true;
                  }
               }
               int warp_done=__any_sync(submask,done);
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
               int WARP=32,
               int elementsPerWarp=1>
      class Hasher{
      
      //Make sure we have sane elements per warp
      static_assert(elementsPerWarp>0 && elementsPerWarp<=WARP && "Device hasher cannot be instantiated");

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
            launchParams(len,blocks,blockSize);

            if constexpr(elementsPerWarp==1){
               hasher_V2<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,Hashinator::defaults::WARPSIZE>
                        <<<blocks,blockSize>>>
                        (src,buckets,sizePower,maxoverflow,d_overflow,d_fill,len);
            }else{
               hasher_V3<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                        <<<blocks,blockSize>>>
                        (src,buckets,sizePower,maxoverflow,d_overflow,d_fill,len);
            }
            cudaDeviceSynchronize();
         }

      private:
         static void launchParams(size_t N,size_t& blocks,size_t& blockSize){
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

/* File:    hashers.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines parallel hashers that insert elements
 *              to Hahsinator on device
 *
 * This file defines the following:
 *    --Hashinator::Hashers::reset_to_empty()
 *    --Hashinator::Hashers::insert
 *    --Hashinator::Hashers::insert_kernel()
 *    --Hashinator::Hashers::retrieve_kernel()
 *    --Hashinator::Hashers::delete_kernel()
 *    --Hashinator::Hashers::insert()
 *    --Hashinator::Hashers::retrieve()
 *    --Hashinator::Hashers::remove()
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
      void reset_to_empty(std::pair<KEY_TYPE, VAL_TYPE>* src,
                          std::pair<KEY_TYPE, VAL_TYPE>* dst,
                          const int sizePower,
                          int maxoverflow,
                          size_t Nsrc)

      {
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         if (tid>=Nsrc){return ;}
         std::pair<KEY_TYPE,VAL_TYPE>&candidate=src[tid];
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
               class HashFunction=HashFunctions::Murmur<KEY_TYPE>,
               int WARPSIZE=32,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(std::pair<KEY_TYPE, VAL_TYPE>* src,
                         std::pair<KEY_TYPE, VAL_TYPE>* buckets,
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

         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         std::pair<KEY_TYPE,VAL_TYPE>&candidate=src[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidate.first,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;

         //Check for duplicates
         for(int i=0; i<(*d_overflow); i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            //If we encounter empty  break as the
            uint32_t mask_already_exists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidate.first)&submask;
            uint32_t emptyFound = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            //If we encountered empty and there is no duplicate in this probing
            //chain we are done.
            if (!mask_already_exists && emptyFound){
               break;
            }
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
            uint32_t mask = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            while(mask){
               int winner =__ffs ( mask ) -1;
               int sub_winner =winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidate.first);
                  if (old == EMPTYBUCKET){
                     int overflow = probingindex-optimalindex;
                     atomicExch(&buckets[probingindex].second,candidate.second);
                     //For some reason this is faster than callign atomicMax without the if
                     if (overflow>*d_overflow){
                        atomicMax((int*)d_overflow,overflow);
                     }
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
               mask ^= (1UL << winner);
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
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Murmur<KEY_TYPE>,
               int WARPSIZE=32,
               int elementsPerWarp>
      __global__ 
      void insert_kernel(KEY_TYPE* keys,
                         VAL_TYPE* vals,
                         std::pair<KEY_TYPE, VAL_TYPE>* buckets,
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

         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         KEY_TYPE& candidateKey=keys[wid];
         VAL_TYPE& candidateVal=vals[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidateKey,sizePower);
         const size_t optimalindex=(hashIndex) & bitMask;

         //Check for duplicates
         for(int i=0; i<(*d_overflow); i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            //If we encounter empty  break as the
            uint32_t mask_already_exists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidateKey)&submask;
            uint32_t emptyFound = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            //If we encountered empty and there is no duplicate in this probing
            //chain we are done.
            if (!mask_already_exists && emptyFound){
               break;
            }
            if (mask_already_exists){
               int winner =__ffs ( mask_already_exists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  atomicExch(&buckets[probingindex].second,candidateVal);
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
            uint32_t mask = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            while(mask){
               int winner =__ffs ( mask ) -1;
               int sub_winner=winner-(subwarp_relative_index)*VIRTUALWARP;
               if (w_tid==sub_winner){
                  KEY_TYPE old = atomicCAS(&buckets[probingindex].first, EMPTYBUCKET, candidateKey);
                  if (old == EMPTYBUCKET){
                     int overflow = probingindex-optimalindex;
                     atomicExch(&buckets[probingindex].second,candidateVal);
                     //For some reason this is faster than callign atomicMax without the if
                     if (overflow>*d_overflow){
                        atomicMax((int*)d_overflow,overflow);
                     }
                     atomicAdd((unsigned long long int*)d_fill, 1);
                     done=true;
                  }
                  //Parallel stuff are fun. Major edge case!
                  if (old==candidateKey){
                     atomicExch(&buckets[probingindex].second,candidateVal);
                     done=true;
                  }
               }
               int warp_done=__any_sync(submask,done);
               if(warp_done>0){
                  return;
               }
               mask ^= (1UL << winner);
            }
         }
         assert(false && "Hashmap completely overflown");
      }

      /*
       * Similarly to the insert_kernel we examine elements in keys and return their value in vals,
       * if the do exist in the hashmap. If the elements is not found and invalid key is returned;
       * */
      template<typename KEY_TYPE,
               typename VAL_TYPE,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               class HashFunction=HashFunctions::Murmur<KEY_TYPE>,
               int WARPSIZE=32,
               int elementsPerWarp>
      __global__ 
      void retrieve_kernel(KEY_TYPE* keys,
                           VAL_TYPE* vals,
                           std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                           int sizePower,
                           int maxoverflow)
      {

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         unsigned int subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         
         auto getIntraWarpMask = [](unsigned int n ,unsigned int l ,unsigned int r)->unsigned int{
            int num = ((1<<r)-1)^((1<<(l-1))-1);
            return (n^num);
         };

         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         KEY_TYPE& candidateKey=keys[wid];
         VAL_TYPE& candidateVal=vals[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidateKey,sizePower);

         //Check for duplicates
         for(int i=0; i<maxoverflow; i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            uint32_t maskExists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidateKey)&submask;
            uint32_t emptyFound = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            //If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
            if (!maskExists && emptyFound){
               return;
            }
            if (maskExists){
               int winner =__ffs ( maskExists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  atomicExch(&candidateVal,buckets[probingindex].second);
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
               class HashFunction=HashFunctions::Murmur<KEY_TYPE>,
               int WARPSIZE=32,
               int elementsPerWarp>
      __global__ 
      void delete_kernel(KEY_TYPE* keys,
                           std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                           size_t* d_tombstoneCounter,
                           int sizePower,
                           int maxoverflow)
      {

         const int VIRTUALWARP=WARPSIZE/elementsPerWarp;
         const size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
         const size_t wid = tid/VIRTUALWARP;
         const size_t w_tid=tid%VIRTUALWARP;
         unsigned int subwarp_relative_index=(wid)%(WARPSIZE/VIRTUALWARP);
         
         auto getIntraWarpMask = [](unsigned int n ,unsigned int l ,unsigned int r)->unsigned int{
            int num = ((1<<r)-1)^((1<<(l-1))-1);
            return (n^num);
         };

         uint32_t submask;
         if constexpr(elementsPerWarp==1){
            //TODO mind AMD 64 thread wavefronts
            submask=SPLIT_VOTING_MASK;
         }else{
            submask=getIntraWarpMask(0,VIRTUALWARP*subwarp_relative_index+1,VIRTUALWARP*subwarp_relative_index+VIRTUALWARP);
         }
         KEY_TYPE& candidateKey=keys[wid];
         const int bitMask = (1 <<(sizePower )) - 1; 
         const size_t hashIndex = HashFunction::_hash(candidateKey,sizePower);

         //Check for duplicates
         for(int i=0; i<maxoverflow; i+=VIRTUALWARP){
            
            //Get the position we should be looking into
            size_t probingindex=((hashIndex+i+w_tid) & bitMask ) ;
            uint32_t maskExists = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==candidateKey)&submask;
            uint32_t emptyFound = __ballot_sync(SPLIT_VOTING_MASK,buckets[probingindex].first==EMPTYBUCKET)&submask;
            //If we encountered empty and the key is not in the range of this warp that means the key is not in hashmap.
            if (!maskExists && emptyFound){
               return;
            }
            if (maskExists){
               int winner =__ffs ( maskExists ) -1;
               winner-=(subwarp_relative_index)*VIRTUALWARP;
               if(w_tid==winner){
                  atomicExch(&buckets[probingindex].second, TOMBSTONE);
                  atomicAdd((unsigned long long int*)d_tombstoneCounter, 1);
               }
               return;
             }
         }
      }



      template<typename KEY_TYPE,
               typename VAL_TYPE,
               class HashFunction,
               KEY_TYPE EMPTYBUCKET=std::numeric_limits<KEY_TYPE>::max(),
               int WARP=32,
               int elementsPerWarp=1,
               cudaStream_t s = cudaStream_t()>
      class Hasher{
      
      //Make sure we have sane elements per warp
      static_assert(elementsPerWarp>0 && elementsPerWarp<=WARP && "Device hasher cannot be instantiated");

      public:

         //Overload with separate input for keys and values.
         static void insert(KEY_TYPE* keys,
                            VAL_TYPE* vals,
                            std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                            int sizePower,
                            int maxoverflow,
                            int* d_overflow,
                            size_t* d_fill,
                            size_t len)
         {
            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            insert_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,vals,buckets,sizePower,maxoverflow,d_overflow,d_fill,len);
            cudaDeviceSynchronize();
         }
         
         //Overload with std::pair<key,val> (k,v) inputs
         //Used by the tombstone cleaning method.
         static void insert(std::pair<KEY_TYPE, VAL_TYPE>* src,
                            std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                            int sizePower,
                            int maxoverflow,
                            int* d_overflow,
                            size_t* d_fill,
                            size_t len)
         {
            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            insert_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (src,buckets,sizePower,maxoverflow,d_overflow,d_fill,len);
            cudaDeviceSynchronize();
         }

         //Retrieve wrapper
         static void retrieve(KEY_TYPE* keys,
                              VAL_TYPE* vals,
                              std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                              int sizePower,
                              int maxoverflow,
                              size_t len)
         {

            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            retrieve_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,vals,buckets,sizePower,maxoverflow);
            cudaDeviceSynchronize();

         }

         //Delete wrapper
         static void erase(KEY_TYPE* keys,
                            std::pair<KEY_TYPE, VAL_TYPE>* buckets,
                            size_t* d_tombstoneCounter,
                            int sizePower,
                            int maxoverflow,
                            size_t len)
         {

            size_t blocks,blockSize;
            launchParams(len,blocks,blockSize);
            delete_kernel<KEY_TYPE,VAL_TYPE,EMPTYBUCKET,HashFunction,defaults::WARPSIZE,elementsPerWarp>
                     <<<blocks,blockSize,0,s>>>
                     (keys,buckets,d_tombstoneCounter,sizePower,maxoverflow);
            cudaDeviceSynchronize();

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

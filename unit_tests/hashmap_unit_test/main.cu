#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include <gtest/gtest.h>
#include <random>
#include <algorithm>
#include <limits.h>

#define BLOCKSIZE 32
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
constexpr int MINPOWER = 10;
constexpr int MAXPOWER = 11;


using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>> vector ;
typedef split::SplitVector<key_type> ivector ;
typedef Hashmap<key_type,val_type> hashmap;


struct Predicate{
   HASHINATOR_HOSTDEVICE
   inline bool operator()( hash_pair<key_type,val_type>& element)const{
      return element.second%2==0;
   }
};

template <class Fn, class ... Args>
auto execute_and_time(const char* name,Fn fn, Args && ... args) ->bool{
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   bool retval=fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   //std::cout<<name<<" took "<<total_time<<" us"<<std::endl;
   (void)name;
   return retval;
}


void create_input(vector& src, uint32_t bias=0){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<key_type,val_type>& kval=src.at(i);
      kval.first=i + bias;
      kval.second=i;
   }
}


void cpu_write(hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<key_type,val_type>& kval=src.at(i);
      hmap.at(kval.first)=kval.second;
   }
}

__global__ 
void gpu_write(hashmap* hmap, hash_pair<key_type,val_type>*src, size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      hmap->set_element(src[index].first, src[index].second);
   }
}


__global__ 
void gpu_remove_insert(hashmap* hmap, hash_pair<key_type,val_type>*rm,  hash_pair<key_type,val_type>*add, size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index  ==0 ){
      for ( int i =0; i <N ;++i ){
         hash_pair<key_type,val_type>elem=rm[i];
         auto rmval=hmap->read_element(elem.first);
         hmap->device_erase(elem.first);
      }
      for ( int i =0; i <N ;++i ){
         hash_pair<key_type,val_type>elem=add[i];
         hmap->set_element(elem.first,elem.second);
      }
   }
}


__global__
void gpu_delete_even(hashmap* hmap, hash_pair<key_type,val_type>*src,size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=hmap->device_find(src[index].first);
      if (kpos==hmap->device_end()){assert(0 && "Catastrophic crash in deletion");}
      if (kpos->second %2==0 ){
         int retval=hmap->device_erase(kpos->first);
         assert(retval==1 && "Failed to erase!");
         retval=hmap->device_erase(kpos->first);
         assert(retval==0 && "Failed to not  erase!");

      }
   }
   return;
}



__global__
void gpu_recover_odd_elements(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N ){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      key_type key= src[index].first;
      val_type val= src[index].second;
      if (val%2!=0){
         auto it=hmap->device_find(key);
         if (it==hmap->device_end()){
            assert( 0 && "Failed in GPU RECOVER ALL ");
         }
         if (it->first!=key || it->second!=val){
            assert( 0 && "Failed in GPU RECOVER ALL ");
         }
      }
   }

   //Iterate over all elements with 1 thread and check for evens;
   if (index==0){
      for (auto it=hmap->device_begin(); it!=hmap->device_end(); ++it){
         if (it->second%2==0 ){
            printf("Found even when there should not be any!\n");
            assert(0);
         }
      }
   }
   return;
}

bool recover_odd_elements(const hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<key_type,val_type>& kval=src.at(i);
      if (kval.second%2!=0){
         auto retval=hmap.find(kval.first);
         if (retval==hmap.end()){return false;}
         bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
         if (!sane){ 
            return false; 
         }
      }
   }
   return true;
}

bool recover_all_elements(const hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<key_type,val_type>& kval=src.at(i);
      //std::cout<<"Validating "<<kval.first<<std::endl;
      auto retval=hmap.find(kval.first);
      if (retval==hmap.end()){
         std::cout<<"INVALID= "<<kval.first<<std::endl;
         return false;
      }
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   //std::cout<<"Key validated "<<retval->first<<" "<<retval->second<<std::endl;
   }
   return true;
}

bool recover_odd_elements(hashmap* hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<key_type,val_type>& kval=src.at(i);
      if (kval.second%2!=0){
         auto retval=hmap->find(kval.first);
         if (retval==hmap->end()){return false;}
         bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
         if (!sane){ 
            return false; 
         }
      }
   }
   return true;
}

bool recover_all_elements(hashmap* hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<key_type,val_type>& kval=src.at(i);
      auto retval=hmap->find(kval.first);
      if (retval==hmap->end()){return false;}
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   }
   return true;
}

__global__
void gpu_recover_all_elements(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      key_type key= src[index].first;
      val_type val= src[index].second;
      auto it=hmap->device_find(key);
      if (it==hmap->device_end()){
         printf("END FOUND DEVICE\n");
         assert( 0 && "Failed in GPU RECOVER ALL ");
      }
      if (it->first!=key || it->second!=val){
         assert( 0 && "Failed in GPU RECOVER ALL ");
      }
   }
   return;
}
__global__
void gpu_recover_warpWide(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[wid].first;
      val_type retval;;
      val_type val= src[wid].second;
      hmap->warpFind(key,retval,w_tid);
      assert(retval==val);
   }
}

__global__
void gpu_recover_non_existant_key_warpWide(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      val_type retval=42;;
      key_type key=42 ;
      hmap->warpFind(key,retval,w_tid);
      assert(retval==42);
   }
}

__global__
void gpu_write_warpWide(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[wid].first;
      val_type val= src[wid].second;
      hmap->warpInsert(key,val,w_tid);
   }
}

__global__
void gpu_write_warpWide_UnorderedSet(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[wid].first;
      val_type val= src[wid].second;
      hmap->warpInsert<1>(key,val,w_tid);
   }
}

__global__
void gpu_write_warpWide_Duplicate(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[0].first;
      val_type val= src[0].second;
      hmap->warpInsert(key,val,w_tid);
   }
}

__global__
void gpu_erase_warpWide(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[wid].first;
      hmap->warpErase(key,w_tid);
   }
}

__global__
void gpu_write_warpWide_V(hashmap* hmap,hash_pair<key_type,val_type>* src,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key= src[wid].first;
      val_type val= src[wid].second;
      auto retval=hmap->warpInsert_V(key,val,w_tid);
      assert(retval);
   }
}

bool testWarpInsert(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=N/blocksize;
   size_t warpsize     =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded  =  N*warpsize; 
   blocks = threadsNeeded/BLOCKSIZE;
 
   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap* hmap=new hashmap;
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write_warpWide<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(*hmap,src);
   if (!cpuOK){
      return false;
   }

   //duplicate test
   {
      size_t N = 1<<power;
      size_t warpsize     =  Hashinator::defaults::WARPSIZE;
      size_t threadsNeeded  =  N*warpsize; 
      blocks = threadsNeeded/BLOCKSIZE;
      //Create some input data
      vector src(N);
      create_input(src);
      hashmap* hmap=new hashmap;
      hmap->resize(power+1);
      //Upload to device and insert input
      gpu_write_warpWide_Duplicate<<<1,1024>>>(hmap,src.data(),1);
      split_gpuDeviceSynchronize();
      if (hmap->size()!=1){
         return false;
      }
   }

   return true;
}

bool testWarpInsertUnorderedSet(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=N/blocksize;
   size_t warpsize     =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded  =  N*warpsize; 
   blocks = threadsNeeded/BLOCKSIZE;
 
   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap* hmap=new hashmap;
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write_warpWide_UnorderedSet<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(*hmap,src);
   if (!cpuOK){
      return false;
   }

   
   //Now we change the key values and increment them by 1 and we expect the same keys back because we are not supposed to overwrite
   vector src2(src);
   for (auto& i:src2){
      i.second++;
   }

   //Upload to device and insert input
   gpu_write_warpWide_UnorderedSet<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(*hmap,src);
   if (!cpuOK){
      return false;
   }

   //duplicate test
   {
      size_t N = 1<<power;
      size_t warpsize     =  Hashinator::defaults::WARPSIZE;
      size_t threadsNeeded  =  N*warpsize; 
      blocks = threadsNeeded/BLOCKSIZE;
      //Create some input data
      vector src(N);
      create_input(src);
      hashmap* hmap=new hashmap;
      hmap->resize(power+1);
      //Upload to device and insert input
      gpu_write_warpWide_Duplicate<<<1,1024>>>(hmap,src.data(),1);
      split_gpuDeviceSynchronize();
      if (hmap->size()!=1){
         return false;
      }
   }

   return true;
}

bool testWarpInsert_V(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=N/blocksize;
   size_t warpsize     =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded  =  N*warpsize; 
   blocks = threadsNeeded/BLOCKSIZE;
 
   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap* hmap=new hashmap;
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write_warpWide_V<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(*hmap,src);
   if (!cpuOK){
      return false;
   }
   return true;
}

bool testWarpErase(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=N/blocksize;
   size_t warpsize     =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded  =  N*warpsize; 
   blocks = threadsNeeded/BLOCKSIZE;
 

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap* hmap=new hashmap;
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write_warpWide_V<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Upload to device and insert input
   gpu_erase_warpWide<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   if (hmap->size()!=0){
      return false;
   }

   return true;
}

bool testWarpFind(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=N/blocksize;

   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);
   ivector keys_only;
   for (const auto& i:src){
      keys_only.push_back(i.first);
   }
   hashmap* hmap=new hashmap;
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(*hmap,src);
   gpu_recover_all_elements<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   if (!cpuOK){
      return false;
   }
   
   size_t warpsize     =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded  =  N*warpsize; 
   blocks = threadsNeeded/BLOCKSIZE;
   gpu_recover_warpWide<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   hmap->erase(keys_only.data(),keys_only.size());
   gpu_recover_non_existant_key_warpWide<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   return true;

}

bool test_hashmap_1(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=2*N/blocksize;

   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap hmap;
   hashmap* d_hmap;
   hmap.resize(power+1);

   //Upload to device and insert input
   d_hmap=hmap.upload();
   gpu_write<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   hmap.download();

   //Verify all elements
   cpuOK=recover_all_elements(hmap,src);
   gpu_recover_all_elements<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   return true;
   if (!cpuOK){
      return false;
   }

   //Delete some selection of the source data
   d_hmap=hmap.upload();
   gpu_delete_even<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   hmap.download();

   //Quick check to verify there are no even elements
   for (const auto& kval : hmap){
      if (kval.second%2==0){
         std::cout<<kval.first<<" "<<kval.second<<std::endl;
         return false;
      }
   }
   
   //Verify odd elements;
   cpuOK=recover_odd_elements(hmap,src);
   gpu_recover_odd_elements<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   if (!cpuOK){
      return false;
   }

   //Reinsert so that we can also test duplicate insertion
   d_hmap=hmap.upload();
   gpu_write<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   //Download
   hmap.download();


   //Verify all elements
   cpuOK=recover_all_elements(hmap,src);
   gpu_recover_all_elements<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   if (!cpuOK ){
      return false;
   }

   //If we made it to here we should be ok 
   return true;
}


bool test_hashmap_2(int power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=2*N/blocksize;
   bool cpuOK=true;

   //Create some input data
   vector src(N);
   create_input(src);


   hashmap* hmap = new hashmap();
   hmap->resize(power+1);

   //Upload to device and insert input
   gpu_write<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(hmap,src);
   gpu_recover_all_elements<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   if (!cpuOK ){
      return false;
   }

   //Delete some selection of the source data
   gpu_delete_even<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();


   //Upload to device and insert input
   gpu_write<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Upload to device and insert input
   gpu_write<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();


   //Delete some selection of the source data
   gpu_delete_even<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Quick check to verify there are no even elements
   for (const auto& kval : *hmap){
      if (kval.second%2==0){
         std::cout<<kval.first<<" "<<kval.second<<std::endl;
         return false;
      }
   }
   
   //Verify odd elements;
   cpuOK=recover_odd_elements(hmap,src);
   gpu_recover_odd_elements<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   //split_gpuDeviceSynchronize();
   if (!cpuOK){
      return false;
   }

   //Clean Tomstones and reinsert so that we can also test duplicate insertion
   hmap->clean_tombstones();
   gpu_write<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();

   //Verify all elements
   cpuOK=recover_all_elements(hmap,src);
   gpu_recover_all_elements<<<blocks,blocksize>>>(hmap,src.data(),src.size());
   split_gpuDeviceSynchronize();
   if (!cpuOK ){
      return false;
   }

   vector src2(N);
   create_input(src2);
   gpu_remove_insert<<<1,1>>>(hmap,src.data(),src2.data(),src.size());
   split_gpuDeviceSynchronize();
   gpu_recover_all_elements<<<blocks,blocksize>>>(hmap,src2.data(),src2.size());
   split_gpuDeviceSynchronize();

   delete hmap;
   hmap=nullptr;
   return true;
}

bool test_hashmap_3(int power){
   size_t N = 1<<power;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap hmap;
   bool cpuOK;

   for (auto i : src){
      hmap.insert(i);
   }

   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      return false;
   }

   for (auto i:hmap){
      if (i.second%2==0){
         hmap.erase(i.first);
      }
   }

   cpuOK=recover_odd_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering odd elements 2"<<std::endl;
      return false;
   }

   for (auto i : src){
      hmap.insert(i);
   }

   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 2"<<std::endl;
      return false;
   }
   return true;
}


bool test_hashmap_4(int power){
   size_t N = 1<<power;

   //Create some input data
   vector src(N);
   create_input(src);
   hashmap hmap;
   bool cpuOK;

   hmap.insert(src.data(),src.size());

   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      return false;
   }

   //Get all even elements in src
   vector evenBuffer(src.size());
   ivector keyBuffer;
   split::tools::copy_if<hash_pair<key_type, val_type>,Predicate>(src,evenBuffer,Predicate());
   for (auto i:evenBuffer){
      keyBuffer.push_back(i.first);
   }


   //Erase using device
   hmap.erase(keyBuffer.data(),keyBuffer.size());

   cpuOK=recover_odd_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering odd elements 2"<<std::endl;
      return false;
   }

   //Quick check to verify there are no even elements
   for (const auto& kval : hmap){
      if (kval.second%2==0){
         std::cout<<kval.first<<" "<<kval.second<<std::endl;
         return false;
      }
   }

   split_gpuStream_t s ;
   split_gpuStreamCreate(&s);
   hmap.clean_tombstones(s);
   cpuOK=recover_odd_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering odd elements 2"<<std::endl;
      return false;
   }
   hmap.insert(src.data(),src.size());

   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 2"<<std::endl;
      return false;
   }
   return true;
}

TEST(HashmapUnitTets , Test_Construction){
   hashmap map0(12);
   expect_true(map0.size()==0);
   for (key_type i=0 ; i< 1<<11; i++){
      map0[i]=i;
   }
   expect_true(map0.size()==1<<11);
   hashmap map1(map0);
   expect_true(map1.size()==1<<11);
   hashmap map2 = map0;
   expect_true(map2.size()==1<<11);
   hashmap map3(hashmap(12));
   expect_true(map3.size()==0);
   expect_true(map3.bucket_count()==1<<12);
   map3=hashmap(13);
   expect_true(map3.size()==0);
   expect_true(map3.bucket_count()==1<<13);
}



TEST(HashmapUnitTets , Test1_HostDevice_UploadDownload){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),test_hashmap_1 ,power);
      expect_true(retval);
   }
}


TEST(HashmapUnitTets , Test1_HostDevice_WarpFind){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),testWarpFind ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test1_HostDevice_WarpInsert){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),testWarpInsert ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test1_HostDevice_WarpInsertUnorderedSet){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),testWarpInsertUnorderedSet ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test1_HostDevice_WarpInsert_V){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),testWarpInsert_V ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test1_HostDevice_WarpErase){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),testWarpErase ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test2_HostDevice_New_Unified_Ptr){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),test_hashmap_2 ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test3_Host){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),test_hashmap_3 ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets , Test4_DeviceKernels){
   for (int power=MINPOWER; power<MAXPOWER; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),test_hashmap_4 ,power);
      expect_true(retval);
   }
}

TEST(HashmapUnitTets ,Test_Clear_Perf_Host){

   const int sz=22;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap(sz);
   bool cpuOK;
   hmap.insert(src.data(),src.size());
   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      expect_true(false);
   }
   hmap.optimizeGPU();
   split_gpuDeviceSynchronize();
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   start = std::chrono::high_resolution_clock::now();
   hmap.clear();
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   //std::cout<<"Clear took "<<duration<<" us status= "<<hmap.peek_status()<<std::endl;
}

TEST(HashmapUnitTets ,Test_Clear_Perf_Device){

   const int sz=22;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap(sz);
   bool cpuOK;
   hmap.insert(src.data(),src.size());
   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      expect_true(false);
   }
   hmap.optimizeGPU();
   split_gpuDeviceSynchronize();
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   start = std::chrono::high_resolution_clock::now();
   hmap.clear(targets::device);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   //std::cout<<"Clear took "<<duration<<" us status= "<<hmap.peek_status()<<std::endl;
}

TEST(HashmapUnitTets ,Test_Resize_Perf_Host){

   const int sz=24;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap(sz);
   bool cpuOK;
   hmap.insert(src.data(),src.size());
   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      expect_true(false);
   }
   split_gpuDeviceSynchronize();
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   start = std::chrono::high_resolution_clock::now();
   hmap.resize(sz+2);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   //std::cout<<"Resize took "<<duration<<" us status= "<<hmap.peek_status()<<std::endl;
}


TEST(HashmapUnitTets ,Test_Resize_Perf_Device){

   const int sz=24;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap(sz);
   bool cpuOK;
   hmap.insert(src.data(),src.size());
   cpuOK=recover_all_elements(hmap,src);
   if (!cpuOK){
      std::cout<<"Error at recovering all elements 1"<<std::endl;
      expect_true(false);
   }
   split_gpuDeviceSynchronize();
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   start = std::chrono::high_resolution_clock::now();
   hmap.resize(sz+2,targets::device);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   //std::cout<<"Resize took "<<duration<<" us"<<std::endl;
   expect_true(hmap.peek_status()==status::success);
}


template <typename T, typename U>
struct Rule{
Rule(){}
   __host__ __device__
   inline bool operator()( hash_pair<T,U>& element)const{
      return element.first<1000;
   }
};


TEST(HashmapUnitTets ,Test_ErrorCodes_ExtractKeysByPattern){
   const int sz=5;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap;
   hmap.insert(src.data(),src.size());
   bool cpuOK=recover_all_elements(hmap,src);
   expect_true(cpuOK);
   expect_true(hmap.peek_status()==status::success);
   ivector out;
   hmap.extractKeysByPattern(out,Rule<key_type,key_type>());
   for (auto i:out){
      expect_true(i<1000);
   }
}


TEST(HashmapUnitTets ,Test_Copy_Metadata){
   const int sz=18;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap;
   hmap.insert(src.data(),src.size());
   bool cpuOK=recover_all_elements(hmap,src);
   expect_true(cpuOK);
   expect_true(hmap.peek_status()==status::success);
   Info info;
   hmap.copyMetadata(&info);
   split_gpuDeviceSynchronize();
   expect_true(1<<info.sizePower==hmap.bucket_count());
   expect_true(info.tombstoneCounter==hmap.tombstone_count());

}

std::vector<key_type> generateUniqueRandomKeys(size_t size, size_t range=std::numeric_limits<int>::max()) {
    std::vector<key_type> elements;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, range);

    for (size_t i = 0; i < size; ++i) {
        key_type randomNum = i;//dist(gen);
        if (std::find(elements.begin(), elements.end(), randomNum) == elements.end()) {
            elements.push_back(randomNum);
        } else {
            --i;  
        }
    }
    return elements;
}

void insertDuplicates(std::vector<key_type>& vec, key_type element, size_t count) {
   if (count>0){
    vec.insert(vec.end(), count, element);
   }
   srand(time(NULL));
   std::random_shuffle(vec.begin(),vec.end());
}

TEST(HashmapUnitTets ,Test_Duplicate_Insertion){
   const int sz=10;
   for (size_t duplicates=2; duplicates<=(1<<sz);duplicates*=2){
      std::vector<key_type> keys=generateUniqueRandomKeys(1<<sz);

      for (size_t i = 0; i < duplicates;i++){
         insertDuplicates(keys,keys[0],1);
      }

      vector src(keys.size());
      for (size_t i =0;i<keys.size(); i++){
         src[i].first=keys[i];
         src[i].second=keys[i];
      }
      hashmap hmap;
      hmap.insert(src.data(),src.size(),1);
      bool cpuOK=recover_all_elements(hmap,src);
      expect_true(cpuOK);
      expect_true(hmap.peek_status()==status::success);
      expect_true(hmap.size()==((1<<sz)));
   }
}



int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

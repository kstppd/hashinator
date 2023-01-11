#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>

#define BLOCKSIZE 1024
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
typedef uint32_t val_type;
typedef split::SplitVector<hash_pair<val_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
using namespace Hashinator;
using namespace std::chrono;
typedef Hashmap<val_type,val_type> hashmap;


template <class Fn, class ... Args>
auto execute_and_time(const char* name,Fn fn, Args && ... args) ->bool{
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   bool retval=fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   std::cout<<name<<" took "<<total_time<<" us"<<std::endl;
   return retval;
}
 


void create_input(vector& src, uint32_t bias=0){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<val_type,val_type>& kval=src.at(i);
      kval.first=i + bias;
      kval.second=rand()%1000000;
   }
}

void cpu_write(hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<val_type,val_type>& kval=src.at(i);
      hmap.at(kval.first)=kval.second;
   }
}

__global__ 
void gpu_write(hashmap* hmap, hash_pair<val_type,val_type>*src, size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      hmap->set_element(src[index].first, src[index].second);
   }
}

__global__
void gpu_delete_even(hashmap* hmap, hash_pair<val_type,val_type>*src,size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=hmap->find(src[index].first);
      if (kpos==hmap->end()){assert(0 && "Catastrophic crash in deletion");}
      if (kpos->second %2==0 ){
         hmap->erase(kpos);
      }
   }
   return;
}

bool recover_elements(const hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<val_type,val_type>& kval=src.at(i);
      auto retval=hmap.find(kval.first);
      if (retval==hmap.end()){assert(0&& "END FOUND");}
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   }
   return true;
}

bool test_hashmap_1(val_type power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=2*N/blocksize;
   vector src(N);
   create_input(src);

   hashmap hmap;
   hmap.insert(src.data(),src.size(),power);
   assert(recover_elements(hmap,src) && "Hashmap is illformed!");
   return true;
}

TEST(HashmapUnitTets , Device_Insert){
   int reps=1;
   for (int power=20; power<21; ++power){
      std::string name= "Power= "+std::to_string(power);
      for (int i =0; i< reps; i++){
         bool retval = execute_and_time(name.c_str(),test_hashmap_1 ,power);
         expect_true(retval);
      }
   }
}


int main(int argc, char* argv[]){
   //srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

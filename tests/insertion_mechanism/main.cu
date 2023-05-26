#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include "hip/hip_runtime.h"

#include <gtest/gtest.h>

#define BLOCKSIZE 1024
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
typedef uint32_t keyval_type;
using namespace Hashinator;
typedef split::SplitVector<keyval_type,split::split_unified_allocator<keyval_type>,split::split_unified_allocator<size_t>> vector ;
using namespace std::chrono;
typedef Hashmap<keyval_type,keyval_type> hashmap;


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

void fill_input(keyval_type* keys , keyval_type* vals, size_t size){
   for (size_t i=0; i<size; ++i){
      keys[i]=i;
      vals[i]=rand()%1000000;
   }
}

bool recover_elements(const hashmap& hmap, keyval_type* keys, keyval_type* vals,size_t size){
   for (size_t i=0; i<size; ++i){
      const hash_pair<keyval_type,keyval_type> kval(keys[i],vals[i]);
      auto retval=hmap.find(kval.first);
      if (retval==hmap.end()){assert(0&& "END FOUND");}
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   }
   return true;
}

bool test_hashmap_insertionDM(keyval_type power){
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=2*N/blocksize;

   vector keys(N);
   vector vals(N);
   fill_input(keys.data(),vals.data(),N);
   fill_input(keys.data(),vals.data(),N);
   keys.optimizeGPU();
   vals.optimizeGPU();


   keyval_type* dkeys;
   keyval_type* dvals;
   hipMalloc(&dkeys, N*sizeof(keyval_type)); 
   hipMalloc(&dvals, N*sizeof(keyval_type)); 
   hipMemcpy(dkeys,keys.data(),N*sizeof(keyval_type),hipMemcpyHostToDevice);
   hipMemcpy(dvals,vals.data(),N*sizeof(keyval_type),hipMemcpyHostToDevice);

   hashmap hmap;
   hmap.insert(dkeys,dvals,N); 
   assert(recover_elements(hmap,keys.data(),vals.data(),N) && "Hashmap is illformed!");
   hipFree(dkeys);
   hipFree(dvals);
   return true;
}

bool test_hashmap_retrievalUM(keyval_type power){
   size_t N = 1<<power;
   vector keys(N);
   vector vals(N);
   vector vals2(N);
   fill_input(keys.data(),vals.data(),N);
   keys.optimizeGPU();
   vals.optimizeGPU();
   vals2.optimizeGPU();

   hashmap hmap;
   hmap.insert(keys.data(),vals.data(),N); 
   assert(recover_elements(hmap,keys.data(),vals.data(),N) && "Hashmap is illformed!");
   keys.optimizeGPU();
   vals.optimizeGPU();
   vals2.optimizeGPU();
   hipDeviceSynchronize();
   hmap.retrieve(keys.data(),vals2.data(),N);
   assert(recover_elements(hmap,keys.data(),vals2.data(),N) && "Hashmap is illformed!");
   return true;
}


bool test_hashmap_insertionUM(keyval_type power){
   size_t N = 1<<power;
   vector keys(N);
   vector vals(N);
   fill_input(keys.data(),vals.data(),N);
   keys.optimizeGPU();
   vals.optimizeGPU();

   hashmap hmap;
   hmap.insert(keys.data(),vals.data(),N); 
   assert(recover_elements(hmap,keys.data(),vals.data(),N) && "Hashmap is illformed!");
   
   //std::cout<<hmap.load_factor()<<std::endl;
   return true;
}

TEST(HashmapUnitTets , Device_Insert_UM){
   int reps=10;
   for (int power=10; power<20; ++power){
      std::string name= "Power= "+std::to_string(power);
      for (int i =0; i< reps; i++){
         bool retval = execute_and_time(name.c_str(),test_hashmap_insertionUM,power );
         //expect_true(retval);
      }
   }
}


TEST(HashmapUnitTets , Device_Insert_DM){
   int reps=10;
   for (int power=10; power<20; ++power){
      std::string name= "Power= "+std::to_string(power);
      for (int i =0; i< reps; i++){
         bool retval = execute_and_time(name.c_str(),test_hashmap_insertionDM ,power);
         expect_true(retval);
      }
   }
}


TEST(HashmapUnitTets , Device_Retrieve_UM){
   int reps=10;
   for (int power=10; power<20; ++power){
      std::string name= "Power= "+std::to_string(power);
      for (int i =0; i< reps; i++){
         bool retval = execute_and_time(name.c_str(),test_hashmap_retrievalUM ,power);
         expect_true(retval);
      }
   }
}

int main(int argc, char* argv[]){
   //srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

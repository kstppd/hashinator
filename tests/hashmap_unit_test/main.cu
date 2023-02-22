#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include <gtest/gtest.h>
#include <cuda/std/utility>

#define BLOCKSIZE 32
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef split::SplitVector<cuda::std::pair<val_type,val_type>,split::split_unified_allocator<cuda::std::pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
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
      cuda::std::pair<val_type,val_type>& kval=src.at(i);
      kval.first=i + bias;
      kval.second=rand()%1000000;
   }
}

void create_random_input(vector& src){
   for (size_t i=0; i<src.size(); ++i){
      cuda::std::pair<val_type,val_type>& kval=src.at(i);
      kval.first=(rand());
      kval.second=rand()%1000000;
   }
}


void cpu_write(hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const cuda::std::pair<val_type,val_type>& kval=src.at(i);
      hmap.at(kval.first)=kval.second;
   }
}

__global__ 
void gpu_write(hashmap* hmap, cuda::std::pair<val_type,val_type>*src, size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      hmap->set_element(src[index].first, src[index].second);
   }
}

__global__
void gpu_delete_even(hashmap* hmap, cuda::std::pair<val_type,val_type>*src,size_t N){
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=hmap->device_find(src[index].first);
      if (kpos==hmap->device_end()){assert(0 && "Catastrophic crash in deletion");}
      if (kpos->second %2==0 ){
         hmap->device_erase(kpos);
      }
   }
   return;
}

bool recover_elements(const hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const cuda::std::pair<val_type,val_type>& kval=src.at(i);
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


   //Create some input data
   vector src(N);
   create_input(src);
   hashmap hmap;
   hashmap* d_hmap;
   hmap.resize(power+1);


   //Upload to device
   d_hmap=hmap.upload();
   gpu_write<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   cudaDeviceSynchronize();
   //Download
   hmap.download();
   assert(recover_elements(hmap,src) &&" Map not verified");

   //Delete some selection of the source data
   d_hmap=hmap.upload();
   gpu_delete_even<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   cudaDeviceSynchronize();

   //Download
   hmap.download();

   //Quick check to verify there are no even elements
   for (const auto& kval : hmap){
      if (kval.second%2==0){std::cout<<kval.first<<" "<<kval.second<<std::endl;}
      assert(kval.second%2!=0 && "There are even elements leaked into the hashmap");
   }

   //All odd elements should be recoverable
   for (const auto& i:src){
      if (i.second%2!=0){
         auto exists=hmap.find(i.first);
         if (exists==hmap.end()){
            std::cout<<i.first<<" "<<i.second<<std::endl;
         assert(false && "Odd element not there");
         }
      }
   }

   //Reinsert so that we can also test duplicate insertion
   d_hmap=hmap.upload();
   gpu_write<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   cudaDeviceSynchronize();
   //Download
   hmap.download();

   //Recover all elements to make sure that the hashmap actually works
   bool retval=recover_elements(hmap,src);
   return retval;
}

TEST(HashmapUnitTets , Host_Device_Insert_Delete_Global_Tets){
   for (int power=5; power<22; ++power){
      std::string name= "Power= "+std::to_string(power);
      bool retval = execute_and_time(name.c_str(),test_hashmap_1 ,power);
      expect_true(retval);
   }
}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>

#define BLOCKSIZE 32
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
typedef uint32_t val_type;
typedef split::SplitVector<hash_pair<val_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
typedef Hashinator<val_type,val_type> hashmap;


void create_input(vector& src){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<val_type,val_type>& kval=src.at(i);
      kval.first=i;
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
      bool sane=retval->first==kval.first  &&  retval->second== kval.second && retval->offset<32;;
      if (!sane){ return false; }
   }
   return true;
}

bool test_hashmap(val_type power){

   //Settings
   size_t N = 1<<power;
   size_t blocksize=BLOCKSIZE;
   size_t blocks=2*N/blocksize;


   //Create some input data
   vector src(N);
   create_input(src);
   hashmap hmap;
   hashmap* d_hmap;
   hmap.resize(power);


   //Upload to device
   //cpu_write(hmap,src);
   d_hmap=hmap.upload();
   gpu_write<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   cudaDeviceSynchronize();
   //Download
   hmap.download();


   //Delete some selection if the source data
   d_hmap=hmap.upload();
   gpu_delete_even<<<blocks,blocksize>>>(d_hmap,src.data(),src.size());
   cudaDeviceSynchronize();
   //Download
   hmap.download();

   //Quick check to verify there are no even elements
   for (const auto& kval : hmap){
      assert(kval.second%2!=0 && "There are even elements leaked into the hashmap");
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
   for (int power=7; power<24; ++power){
      bool retval=test_hashmap(power);
      expect_true(retval);
   }
}




int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

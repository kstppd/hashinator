#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#define N  64

typedef uint32_t val_type;


__global__
void readMap(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   val_type* element=dmap->read_element(index);
   assert(*element == index);

}


__global__
void writeMap(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   dmap->set_element(index,index);

}

__global__
void addtoMap(Hashinator<val_type,val_type> *dmap){
   int index = (blockIdx.x * blockDim.x + threadIdx.x)+268435456;
    dmap->set_element(index, index);
}

void cpuTest(Hashinator<val_type,val_type>& map,int total_keys=N){
   for (val_type i=0; i<total_keys;i++){
      map[i]=i;
   }
}

void addNelems(Hashinator<val_type,val_type>&map,int numel){
   int threads=32;
   auto initial_lf=map.load_factor();
   int blocks=numel/threads;
   //printf("Adding %i elements  with %i Threads and %i Blocks\n ",(int)numel,(int)threads,(int)blocks);
   Hashinator<val_type,val_type>* dmap = map.upload();
   auto start = std::chrono::high_resolution_clock::now();
   addtoMap<<<blocks,threads>>>(dmap);
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   map.clean_up_after_device(dmap);
   //map.print_all();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   printf("--->TIME: %.3f seconds for %zu elements at a load factor of %f\n", total_time.count() * 1e-9,map.size(),initial_lf);
}
   
   


TEST(CPU_TEST,CPU_Read_Check){
   Hashinator<val_type,val_type> map;
   cpuTest(map);
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
}   

TEST(GPU_TEST,GPU_Read_Check){
   Hashinator<val_type,val_type> map;
   cpuTest(map);
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
   map.print_bank();

   Hashinator<val_type,val_type>* dmap = map.upload();
   readMap<<<2,16>>>(dmap);
   cudaDeviceSynchronize();
   map.clean_up_after_device(dmap);
}   


TEST(GPU_TEST,GPU_Write_Check){
   Hashinator<val_type,val_type> map;
   cpuTest(map,1<<20);
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
   map.print_bank();
   addNelems(map, 1000000);
   
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
   map.print_bank();

}   



int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}


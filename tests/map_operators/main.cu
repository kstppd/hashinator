#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#define N  64

typedef uint32_t val_type;

__global__
void gpu_write_map(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   dmap->set_element(index,index);
   return;
}


void cpu_write_map(Hashinator<val_type,val_type>& map,int total_keys=N){
   for (val_type i=0; i<total_keys;i++){
      map[i]=i;
   }
}


TEST(CPU_TEST,CPU_RW){
   Hashinator<val_type,val_type> map;
   cpu_write_map(map);
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
}   

TEST(GPU_TEST,GPU_RW){
   Hashinator<val_type,val_type> map;
   map.resize(8);
   auto* dmap=map.upload();
   map.print_all();
   gpu_write_map<<<1,128>>>(dmap);
   cudaDeviceSynchronize();
   map.clean_up_after_device(dmap);
   for (auto &kval:map){
      EXPECT_TRUE(kval.first == kval.second);
   }
   map.print_all();
}   


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}


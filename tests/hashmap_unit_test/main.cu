#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#include <random>
#include <limits>
#include "prng.hpp"
#include <unordered_map>

#define POWER 5
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ

typedef uint32_t val_type;
typedef Hashinator<val_type,val_type> hashmap;
//typedef std::unordered_map<val_type,val_type> hashmap;

__global__
void gpu_write_map(hashmap *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   std::pair<val_type,val_type> p{index,index};
   auto ret=dmap->insert(p);
   return;
}

__global__
void gpu_delete_even(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index>5){
      auto kpos=dmap->find(index);
      dmap->erase(kpos);
   }
   return;
}


void insertN(hashmap& map, size_t N){
   prng::Generator gen{42u};  // Custom seed
   const uint64_t bound=1<<30;;
   for (size_t i=0; i<N;i++){
      auto key=gen.uniform(bound);
      auto val=gen.uniform(bound);
      map[key]=val;
      
   }
   return;
}

void deleteN(hashmap& map, size_t N){
   prng::Generator gen{42u};  // Custom seed
   const uint64_t bound=1<<30;;
   for (size_t i=0; i<N;i++){
      auto key=gen.uniform(bound);
      auto val=gen.uniform(bound);
      map.erase(key);
   }
   return;
}
//TEST(Benchmark_CPU, Million){
   
   //size_t totalKeys=1e6;
   //hashmap map;
   //insertN(map,totalKeys);
   //map.print_all();
   //map.clear();
   //map.print_all();
   //insertN(map,totalKeys);
   //map.print_all();
   //deleteN(map,totalKeys);
   //map.print_all();
//}

TEST(Benchmark_GPU, Million){

   hashmap map;
   map.resize(POWER+1);

   int threads=32;
   size_t blocks=(1<<POWER)/threads;
   hashmap* dmap;

   //Upload map to device
   dmap=map.upload();

   //Call a simple kernel that just writes to the map elements based on their index
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   map.download();
   map.print_all();
   map.dump_buckets();
   
   //Delete
   dmap=map.upload();
   gpu_delete_even<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   map.download();
   map.print_all();
   map.dump_buckets();

}

int main(int argc, char** argv){
   
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();

}

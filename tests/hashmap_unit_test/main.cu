#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#include <random>
#include <limits>
#include "prng.hpp"
#include <unordered_map>

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ

typedef uint32_t val_type;
typedef Hashinator<val_type,val_type> hashmap;
//typedef std::unordered_map<val_type,val_type> hashmap;


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
TEST(Benchmark_CPU, Million){
   
   size_t totalKeys=1e6;
   hashmap map;
   insertN(map,totalKeys);
   map.print_all();
   map.clear();
   map.print_all();
   insertN(map,totalKeys);
   map.print_all();
   deleteN(map,totalKeys);
   map.print_all();
}


int main(int argc, char** argv){
   
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();

}

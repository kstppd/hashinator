#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include <gtest/gtest.h>

#define BLOCKSIZE 32
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
constexpr int MINPOWER = 5;
constexpr int MAXPOWER = 20;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;



void create_input(vector& src){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<key_type,val_type>& kval=src.at(i);
      kval.first=i;
      kval.second=i;
   }
}


bool benchInsert(){
   using hashmap= Hashmap<key_type,val_type>;
   const int sz=24;
   vector src(1<<sz);
   create_input(src);
   hashmap hmap(sz+1);
   src.optimizeGPU();
   hmap.optimizeGPU();
   hmap.insert(src.data(),src.size());
   return(hmap.peek_status()==status::success);
}

TEST(HashmapUnitTets ,Test_Copy_Metadata){
   for (int i =0; i<10; i++){
      expect_true(benchInsert());
   }
}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

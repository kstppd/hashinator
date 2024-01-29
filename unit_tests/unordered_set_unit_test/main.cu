#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/unordered_set/unordered_set.h"
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


TEST(Unordered_Set_UnitTest , Test1){
      expect_true(true);
}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

#include <iostream>
#include "../../include/hashinator/unordered_set/unordered_set.h"
#include <gtest/gtest.h>

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ

using namespace Hashinator;
typedef uint32_t key_type;
typedef split::SplitVector<key_type> vector ;
typedef Unordered_Set<key_type> Set;


TEST(Unordered_Set_UnitTest , Test1){
   Set s;
   expect_true(true);
}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

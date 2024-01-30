#include <iostream>
#include "../../include/hashinator/unordered_set/unordered_set.h"
#include <gtest/gtest.h>
#include <unordered_set>

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define SMALL_SIZE 512
#define LARGE_SIZE ( 1<<24 )

using namespace Hashinator;
typedef uint32_t key_type;
typedef split::SplitVector<key_type> vector ;
typedef Unordered_Set<key_type> UnorderedSet;


TEST(Unordered_UnitTest , Construction){
   UnorderedSet s(12);
   UnorderedSet s2=s;
   expect_true(s2.bucket_count()==s.bucket_count());
   UnorderedSet s3= UnorderedSet(12);
   expect_true(s3.bucket_count()==1<<12);
   expect_true(true);
}

TEST(Unordered_UnitTest , Construction_InitializerList){
   UnorderedSet s{std::initializer_list<key_type>{1,2,3,4,1,2,3,4}};
   expect_true(s.size()==4);
}

TEST(Unordered_UnitTest , Empty){
   UnorderedSet s;
   expect_true(s.size()==0);
   expect_true(s.empty());
   s.insert(1);
   expect_false(s.empty());
}

TEST(Unordered_UnitTest , InsertFindHost){
   UnorderedSet s;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      s.insert(i);
      auto it = s.find(i);
      expect_true(*it==i);
   }
   expect_true(s.size()==SMALL_SIZE);
   expect_true(s.tombstone_count()==0);
}

TEST(Unordered_Timings , InsertHost){
   UnorderedSet s;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      s.insert(i);
   }
   expect_true(s.size()==LARGE_SIZE);
}

TEST(Unordered_TimingsStd , InsertHost){
   std::unordered_set<key_type> s;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      s.insert(i);
   }
   expect_true(s.size()==LARGE_SIZE);
}

TEST(Unordered_Timings , InsertKernel){
   vector v;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      v.push_back(i);
   }

   UnorderedSet s;
   s.insert_fast(v.data(),v.size());

}

TEST(Unordered_UnitTest , Contains_Count){
   UnorderedSet s;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      s.insert(i);
      expect_true(s.contains(i));
      expect_false(s.contains(SMALL_SIZE+i));
      expect_true(s.count(i)==1);
   }
}

TEST(Unordered_UnitTest , InsertEraseHost){
   UnorderedSet s;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      s.insert(i);
   }
   expect_true(s.size()==SMALL_SIZE);
   expect_true(s.tombstone_count()==0);

   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      auto it = s.find(i);
      expect_true(*it==i);
      s.erase(it);
      auto it2 = s.find(i);
      expect_true(it2==s.end());
   }
   expect_true(s.size()==0);
   expect_true(s.tombstone_count()==SMALL_SIZE);

   s.rehash();
   expect_true(s.size()==0);
   expect_true(s.tombstone_count()==0);

   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      s.insert(i);
      auto it = s.find(i);
      expect_true(*it==i);
   }
   expect_true(s.size()==SMALL_SIZE);
   expect_true(s.tombstone_count()==0);
}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

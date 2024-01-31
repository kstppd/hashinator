#include <iostream>

#define SPLIT_CPU_ONLY_MODE
#define HASHINATOR_CPU_ONLY_MODE

#include "../../include/hashinator/unordered_set/unordered_set.h"
#include <gtest/gtest.h>
#include <unordered_set>

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define SMALL_SIZE (1<<10)
#define LARGE_SIZE ( 1<<20 )

using namespace Hashinator;
typedef uint32_t key_type;
typedef split::SplitVector<key_type> vector ;
typedef Unordered_Set<key_type> UnorderedSet;


bool isFreeOfDuplicates(const vector& v){

   for (const auto & it : v){
      auto cnt = std::count( v.begin() ,v.end(),it);
      if (cnt>1 ){return false;}
   }
   return true ;
}

bool isFreeOfDuplicates( UnorderedSet* s){
   vector out;
   for (auto& k:*s ){
      out.push_back(k);
   }
   expect_true(out.size()==s->size());
   expect_true( isFreeOfDuplicates(out));
   return true;
}
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

TEST(Unordered_UnitTest , InsertHost){
   UnorderedSet s;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      s.insert(i);
   }
   expect_true(s.size()==LARGE_SIZE);
}


TEST(Unordered_UnitTest , InsertKernel){
   vector v;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet s;
   s.insert(v.data(),v.size()) ;
   expect_true(s.size()==LARGE_SIZE);
}


TEST(Unordered_UnitTest , InsertEraseHost){
   std::unordered_set<key_type> s;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      s.insert(i);
   }
   expect_true(s.size()==LARGE_SIZE);
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      s.erase(i);
   }
   expect_true(s.size()==0);
}

TEST(Unordered_UnitTest , Insert_Erase_Kernel){
   vector v;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet s;
   s.insert(v.data(),v.size()) ;
   expect_true(s.size()==LARGE_SIZE);
   s.erase(v.data(),v.size()) ;
   expect_true(s.size()==0);
}

TEST(Unordered_UnitTest , NewOverloadManagedMemory){
   vector v;
   for (uint32_t i = 0 ; i < LARGE_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet* s = new UnorderedSet;
   s->insert(v.data(),v.size()) ;
   expect_true(s->size()==LARGE_SIZE);
   s->erase(v.data(),v.size()) ;
   expect_true(s->size()==0);
   delete s;
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

TEST(Unordered_UnitTest , InsertEraseHostSmall){
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


TEST(Unordered_UnitTest , Clear){
   {
      vector v;
      for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
         v.push_back(i);
      }
      UnorderedSet* s = new UnorderedSet;
      s->insert(v.data(),v.size()) ;
      expect_true(s->size()==SMALL_SIZE);

      s->clear();
      expect_true(s->size()==0);
      expect_true(s->empty());
      delete s;
   }


   {
      vector v;
      for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
         v.push_back(i);
      }
      UnorderedSet* s = new UnorderedSet;
      s->insert(v.data(),v.size()) ;
      expect_true(s->size()==SMALL_SIZE);

      s->clear(targets::device);
      expect_true(s->size()==0);
      expect_true(s->empty());
      delete s;
   }
}

TEST(Unordered_UnitTest , Resize){
   vector v;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet* s = new UnorderedSet;
   s->insert(v.data(),v.size()) ;
   expect_true(s->size()==SMALL_SIZE);
   auto priorFill=s->size();

   auto sizePower = s->getSizePower();
   s->resize(sizePower+1,targets::host);
   expect_true(s->size()==priorFill);
}

TEST(Unordered_UnitTest , LoadFactorReduction){

   vector v;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet* s = new UnorderedSet;
   s->insert(v.data(),v.size(),1.0) ;
   expect_true(s->size()==SMALL_SIZE);

   //At this point we are heavilly overflown 
   expect_true( isFreeOfDuplicates(s) );

   //Let's resize to get back to a proper overflow
   s->performCleanupTasks();
   expect_true( isFreeOfDuplicates(s) );
   delete s;
}

TEST(Unordered_UnitTest , TombstoneCleaning){

   vector v;
   for (uint32_t i = 0 ; i < SMALL_SIZE;++i){
      v.push_back(i);
   }
   UnorderedSet* s = new UnorderedSet;
   s->insert(v.data(),v.size(),1.0) ;
   expect_true(s->size()==SMALL_SIZE);

   //At this point we are heavilly overflown 
   expect_true( isFreeOfDuplicates(s) );

   //Let's resize to get back to a proper overflow
   s->erase(v.data(),v.size()/2);
   expect_true( s->tombstone_count()==SMALL_SIZE/2);
   expect_true( isFreeOfDuplicates(s) );
   expect_true( s->tombstone_count()==SMALL_SIZE/2);
   s->performCleanupTasks();
   expect_true( s->tombstone_count()==0);
   delete s;
}


int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

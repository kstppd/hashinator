#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>



typedef split::SplitVector<int> vec ;
typedef split::SplitVector<split::SplitVector<int>> vec2d ;

TEST(Ctor,Vec_of_Vec){

   vec inner_a(100,1);
   vec inner_b(100,2);
   vec2d a(10,inner_a);
   vec2d b(10,inner_b);


   for (auto &i:a){
      for (const auto &val:i){
         EXPECT_EQ(val, 1);
      }
   }
   
   for (auto &i:b){
      for (const auto &val:i){
         EXPECT_EQ(val, 2);
      }
   }

   b.swap(a);
   EXPECT_FALSE(a==b);
   //a.swap(b);
   EXPECT_FALSE(a==b);


}

TEST(Constructor,Elemtent_and_Size_Checks){
   const int val=42;
   const int size=1e6;
   vec a(size,val);
   for (int i=0; i<a.size(); i++){
      EXPECT_EQ(val, a.at(i));
      EXPECT_EQ(val, a[i]);
   }
   EXPECT_EQ(size, a.size());

}

TEST(Constructor, Copy){
   vec a(1e6,2);

   EXPECT_TRUE(a.size() == 1e6);
   for (int i=0; i<a.size(); i++){
      EXPECT_EQ(2, a.at(i));
   }
   vec b=a;
   EXPECT_TRUE(a == b);
   EXPECT_TRUE(a.data() == b.data());
}

TEST(Constructor, Initializer_List){
   std::initializer_list<int> list({ 1, 2, 3, 4 });
   vec a(list);
   EXPECT_TRUE(a[0] == 1);
   EXPECT_TRUE(a[1] == 2);
   EXPECT_TRUE(a[2] == 3);
   EXPECT_TRUE(a[3] == 4);
   EXPECT_TRUE(a.size() == list.size());

}

TEST(Constructor, Initializer_List_Equality){
   vec a({1,2,3,4});
   vec b({1,2,3,4});
   vec c({-1,2,3,4});
   EXPECT_TRUE(a==b);
   EXPECT_TRUE(b==a);
   EXPECT_FALSE(c==a);
   EXPECT_FALSE(c==b);
   EXPECT_FALSE(a==c);
   EXPECT_FALSE(b==c);

}

TEST(Iterators, Access){
   vec a(10,2);

}


TEST(Swap, Swap){
   vec a(1e6,2);
   vec b(1e4,3);
   EXPECT_TRUE(a.size() == 1e6);
   EXPECT_TRUE(b.size() == 1e4);
   for (int i=0; i<a.size(); i++){
      EXPECT_EQ(2, a.at(i));
   }
   for (int i=0; i<b.size(); i++){
      EXPECT_EQ(3, b.at(i));
   }

   a.swap(b);
   
   EXPECT_TRUE(a.size() == 1e4);
   EXPECT_TRUE(b.size() == 1e6);
   for (int i=0; i<a.size(); i++){
      EXPECT_EQ(3, a.at(i));
   }
   for (int i=0; i<b.size(); i++){
      EXPECT_EQ(2, b.at(i));
   }
}


TEST(Copy, Equal){
   vec a(1e6,2);
   vec b(1e4,3);

   EXPECT_TRUE(a.size() == 1e6);
   EXPECT_TRUE(b.size() == 1e4);
   for (int i=0; i<a.size(); i++){
      EXPECT_EQ(2, a.at(i));
   }
   for (int i=0; i<b.size(); i++){
      EXPECT_EQ(3, b.at(i));
   }

   a=b;

   EXPECT_TRUE(a.size() == b.size());
   EXPECT_TRUE(a == b);
   EXPECT_FALSE(a.data() == b.data());
}

TEST(Operator, Equal){
   vec a(10,2);
   vec b(10,2);
   EXPECT_TRUE(a == b);
}

TEST(Operator, NotEqual){
   vec a(10,2);
   vec b(10,2);
   EXPECT_FALSE(a != b);
}

TEST(Iterators, NotEqual){
   //const vec a(10,2);

   const split::SplitVector< int> a(10,2) ;
   //vec b(10,2);
   split::SplitVector< int>::const_iterator it=a.begin();
   std::cout<<*it<<std::endl;
   *it=3;
   std::cout<<*it<<std::endl;
   //EXPECT_FALSE(a != b);
}



int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

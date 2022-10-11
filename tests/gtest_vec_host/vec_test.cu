#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define N 1<<12

typedef split::SplitVector<int> vec ;
typedef split::SplitVector<split::SplitVector<int>> vec2d ;
typedef split::SplitVector<int>::iterator   split_iterator;



void print_vec_elements(vec& v){
   std::cout<<"****Vector Contents********"<<std::endl;
   std::cout<<"Size= "<<v.size()<<std::endl;
   std::cout<<"Capacity= "<<v.capacity()<<std::endl;
   for (auto i:v){
      std::cout<<i<<std::endl;
   }
   std::cout<<"****~Vector Contents********"<<std::endl;
}

TEST(Constructors,Move){
   vec b(vec(N,2));
   for (size_t i=0 ; i<N; ++i){
      expect_true(b[i]=2);
   }
}


TEST(Test_2D_Contruct,VecOfVec){

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
   expect_true(a!=b);
   expect_true(a!=b);
   expect_false(a==b);
   expect_false(a==b);
}

TEST(Constructors,Default){
   vec a;
   expect_true(a.size()==0 && a.capacity()==0);
   expect_true(a.data()==nullptr);
}

TEST(Constructors,Size_based){
   vec a(N);
   expect_true(a.size()==N && a.capacity()==N);
   expect_true(a.data()!=nullptr);
}

TEST(Constructors,std_vector){
   std::vector<int>  stdvec(N,10);
   vec a(stdvec);

   for (size_t i=0; i<N; i++){
      expect_true(stdvec[i]=a[i]);
   }
   vec b(a);
   expect_true(a==b);
}

TEST(Constructors,Specific_Value){
   vec a(N,5);
   expect_true(a.size()==N && a.capacity()==N);
   for (size_t i=0; i<N;i++){
      expect_true(a[i]==5);
      expect_true(a.at(i)==5);
   }
}

TEST(Constructors,Copy){
   vec a(N,5);
   vec b(a);
   for (size_t i=0; i<N;i++){
      expect_true(a[i]==b[i]);
      expect_true(a.at(i)==b.at(i));
   }
}

TEST(Vector_Functionality , Reserve){
   vec a;
   size_t cap =1000000;
   a.reserve(cap);
   expect_true(a.size()==0);
   expect_true(a.capacity()==cap);
}

TEST(Vector_Functionality , Reserve2){

   for (int i =1; i<100; i++){
      vec a(N,i);
      vec b(a);

      size_t cap =32*N;
      a.reserve(cap);
      expect_true(a==b);
   }
}

TEST(Vector_Functionality , Resize){
   vec a;
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   expect_true(a.capacity()==a.size());
}

TEST(Vector_Functionality , Swap){
   vec a(10,2),b(10,2);
   a.swap(b);
   expect_true(a==b);
   vec c(100,1);
   vec d (200,3);
   c.swap(d);
   expect_true(c.size()==200);
   expect_true(d.size()==100);
   expect_true(c.front()==3);
   expect_true(d.front()==1);

}

TEST(Vector_Functionality , Resize2){
   vec a;
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   expect_true(a.capacity()==a.size());
}

TEST(Vector_Functionality , Clear){
   vec a(10);
   size_t size =1<<20;
   a.resize(size);
   expect_true(a.size()==size);
   auto cap=a.capacity();
   a.clear();
   expect_true(a.size()==0);
   expect_true(a.capacity()==cap);
}

TEST(Vector_Functionality , PopBack){
   vec a{1,2,3,4,5,6,7,8,9,10};
   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();
   for (int i=9;i>=0;i--){
      a.pop_back();
      if (a.size()>0){
         expect_true(i==a.back());
      }
   }
   expect_true(a.size()==0);
   expect_true(a.capacity()==initial_cap);
}

TEST(Vector_Functionality , Push_Back){
   vec a;
   for (auto i=a.begin(); i!=a.end();i++){
      expect_true(false);
   }

   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   a.push_back(11);
   expect_true(11==a[a.size()-1]);
   a.push_back(12);
   expect_true(12==a[a.size()-1]);

}


TEST(Vector_Functionality , Shrink_to_Fit){
   vec a;
   for (auto i=a.begin(); i!=a.end();i++){
      expect_true(false);
   }

   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   for (int i =0 ; i< 1024; i++){
      a.push_back(i);
   }

   expect_true(a.size()<a.capacity());
   a.shrink_to_fit();
   expect_true(a.size()==a.capacity());

}
TEST(Vector_Functionality , Push_Back_2){
   vec a{1,2,3,4,5,6,7,8,9,10};
   size_t initial_size=a.size();
   size_t initial_cap=a.capacity();

   a.push_back(11);
   expect_true(11==a[a.size()-1]);
   a.push_back(12);
   expect_true(12==a[a.size()-1]);

}

TEST(Vector_Functionality , Insert_1_Element){
   {
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      auto it(a.begin());
      auto it2=a.insert(it,-1);
      expect_true(a[0]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(a.end());
      auto it2=a.insert(it,-1);
      expect_true(a.back()=-1);
      expect_true(a[a.size()-1]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(&a[4]);
      auto it2=a.insert(it,-1);
      expect_true(a[4]=-1);
      expect_true(a.size()==s0+1);
      expect_true(a.capacity()>c0);
   }
   {
     vec a{1,2,3,4,5,6,7,8,9,10};
     auto s0=a.size(); auto c0=a.capacity();
     try {
      //hehe
      vec::iterator it(nullptr);
      auto it2=a.insert(it,-1);
     }// this has to throw
     catch (...) {
        expect_true(true);
        expect_true(a.capacity()==c0);
        expect_true(a.size()==s0);
           return;
     }
     //if we end up here it never threw so something's up
     expect_true(false);
   }
}


TEST(Vector_Functionality , Insert_Many_Elements){

   {
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      auto it(a.begin());
      auto it2=a.insert(it,10,-1);
      for (int i =0 ; i<10 ; i++){
         expect_true(a[i]=-1);
      }
      expect_true(a.front()==-1);
      expect_true(a.size()==s0+10);
   }
   {
      vec a{1,2,3,4,5,6,7,8,9,10};
      auto s0=a.size(); auto c0=a.capacity();
      vec::iterator it(a.end());
      auto it2=a.insert(it,10,-1);
      for (int i =s0 ; i<a.size() ; i++){
         expect_true(a[i]=-1);
      }
      expect_true(a.back()=-1);
      expect_true(a.size()==s0+10);
   }

   {
     vec a{1,2,3,4,5,6,7,8,9,10};
     auto s0=a.size(); auto c0=a.capacity();
     try {
      //hehe
      vec::iterator it(nullptr);
      auto it2=a.insert(it,10,-1);
     }// this has to throw
     catch (...) {
        expect_true(true);
        expect_true(a.capacity()==c0);
        expect_true(a.size()==s0);
           return;
     }
     //if we end up here it never threw so something's up
     expect_true(false);
   }
}


TEST(Vector_Functionality , Insert_Range_Based){

   {
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10};
      auto backup(a);
      split::SplitVector<int> b{-1,-2,-3,-4,-5,-6,-7,-8,-9,-10};
      auto s0=a.size();
      auto it(a.end());
      auto it_b0(b.begin());
      auto it_b1(b.end());
      a.insert(it,it_b0,it_b1);
      expect_true(a.size()==s0+b.size());
      for (int i=0 ; i <10 ; i++){
         expect_true(a[i]=backup[i]);

      }
      for (int i=10 ; i <20 ; i++){
         expect_true(a[i]=b[i-10]);
      }
   }


   {
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10};
      auto backup(a);
      split::SplitVector<int> b{-1,-2,-3,-4,-5,-6,-7,-8,-9,-10};
      auto s0=a.size();
      auto it(a.end());
      auto it_b0(b.begin());
      auto it_b1(b.begin());
      a.insert(it,it_b0,it_b1);
      expect_true(a.size()==s0);
      for (int i=0 ; i <10 ; i++){
         expect_true(a[i]=backup[i]);

      }
   }

}


TEST(Vector_Functionality , Erase_Single){
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10};
      split::SplitVector<int> b{1,2,3,5,6,7,8,9,10};
      split_iterator it0=&a[3];
      a.erase(it0);
      expect_true(a==b);
}
TEST(Vector_Functionality , Erase_Range){
      split::SplitVector<int> a{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
      split::SplitVector<int> b{1,2,3,4,9,10,11,12,13,14,15};
      split_iterator it0=&a[3];
      split_iterator it1=&a[7];
      a.erase(it0,it1);
      expect_true(a==b);
}

//TEST(Vector_Functionality , Emplace_Back){
   //vec a;
   //for (auto i=a.begin(); i!=a.end();i++){
      //expect_true(false);
   //}
   //size_t initial_size=a.size();
   //size_t initial_cap=a.capacity();

   //std::cout<<"--------------------Before\n";
   //a.emplace_back(11);
   //std::cout<<"--------------------After\n";
   ////expect_true(11==a[a.size()-1]);
   ////a.emplace_back(12);
   ////expect_true(12==a[a.size()-1]);

//}

////TEST(Vector_Functionality , Emplace_Back_2){
   //vec a{1,2,3,4,5,6,7,8,9,10};
   //size_t initial_size=a.size();
   //size_t initial_cap=a.capacity();
   //a.emplace_back(11);
   //expect_true(11==a[a.size()-1]);
   //a.emplace_back(12);
   //expect_true(12==a[a.size()-1]);
//}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

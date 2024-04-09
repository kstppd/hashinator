#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"
#include "include/common.h"
#include "include/splitvector/archMacros.h"
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define TARGET 1

typedef uint32_t int_type ;
typedef struct{
   int_type num;
   int_type flag;
} test_t;
typedef split::SplitVector<test_t> vector; 
size_t count = 0;

void print_vector(vector& v){
   std::cout<<"-------------------"<<std::endl;
   std::cout<<"Size = "<<v.size()<<std::endl;;
   for (const auto& i:v){
      std::cout<<"["<<i.num<<","<<i.flag<<"] ";
   }
   std::cout<<"\n-------------------"<<std::endl;
   std::cout<<std::endl;
}

void fill_vec(vector& v, size_t targetSize){
   count=0;
   size_t st=0;
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<int_type> dist(1, std::numeric_limits<int_type>::max());
   v.clear();
   while (v.size() < targetSize) {
      int_type val =++st;// dist(gen);
      v.push_back(test_t{val,(val%2==0)});
      if (val%2 == 0){count++;};
    }
}

void fill_vec_lin(vector& v, size_t targetSize){
   v.clear();
   int_type s=0;
   while (v.size() < targetSize) {
      v.push_back(test_t{s,s});
      s++;
    }
}

bool checkFlags(const vector& v,const int_type target){
   for (const auto& i:v){
      if (i.flag!=target){return false;}
   }
   return true;
}

bool run_test(int power){
   //std::cout<<"Testing with vector size: "<<( 1<<power )<<std::endl;
   vector v;
   fill_vec(v,1<<power);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
   split::tools::copy_if(v,output1,predicate_on);
   split::tools::copy_if(v,output2,predicate_off);
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   return sane1 && sane2 && sane3 && sane4 && sane5;
}

bool run_test_small(size_t size){
   //std::cout<<"Testing with vector size: "<<size<<std::endl;
   vector v;
   fill_vec(v,size);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
#if 1
   split::tools::copy_if(v,output1,predicate_on);
   split::tools::copy_if(v,output2,predicate_off);
#else
   auto one = split::tools::copy_if_raw(v,output1.data(),predicate_on);
   auto two = split::tools::copy_if_raw(v,output2.data(),predicate_off);
   output1.erase(&output1[one] , output1.end() );   
   output2.erase(&output2[two] , output2.end() ) ;  
#endif
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   //printf( " %d - %d - %d - %d - %d\n",sane1,sane2,sane3,sane4,sane5 );  
   bool retval =  sane1 && sane2 && sane3 && sane4 && sane5;
   return retval;
}

bool run_test_small_loop_variant(size_t size){
   // std::cout<<"Testing with vector size: "<<size<<std::endl;
   vector* v=new vector();
   fill_vec(*v,size);

   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector* output1=new vector(nextPow2(2*v->size()));
   vector* output2=new vector(nextPow2(2*v->size()));

   split::tools::copy_if_loop(*v,*output1,predicate_on);
   split::tools::copy_if_loop(*v,*output2,predicate_off);
   split_gpuDeviceSynchronize();

   bool sane1 = checkFlags(*output1,1);
   bool sane2 = checkFlags(*output2,0);
   bool sane3 = ((output1->size()+output2->size())==v->size());
   bool sane4 =(  output1->size() ==count );
   bool sane5 = ( output2->size() ==v->size()-count );
   // printf( " %d - %d - %d - %d - %d\n",sane1,sane2,sane3,sane4,sane5 );  
   bool retval =  sane1 && sane2 && sane3 && sane4 && sane5;
   return retval;
}

TEST(StremCompaction , Compaction_Tests_Linear){
   for (size_t s=32; s< 3000; s++ ){
      bool a = run_test_small(s);
      expect_true(a);
   }

}

TEST(StremCompaction , Compaction_Tests_Linear_Loop_Variant){
   for (size_t s=32; s< 1024; s++ ){
      bool a = run_test_small_loop_variant(s);
      expect_true(a);
   }

}

TEST(StremCompaction , Compaction_Tests_Power_of_2){
   for (uint32_t i =5; i< 25; i++){
      expect_true(run_test(i));
   }
}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

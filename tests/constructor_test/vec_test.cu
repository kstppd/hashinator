#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <gtest/gtest.h>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>



typedef split::SplitVector<int> Vec ;
typedef split::SplitVector<split::SplitVector<int>> Vec2d ;

void test_function_val(Vec v){
   std::cout<<"val function call"<<std::endl;
}

void test_function_ref(Vec &v){
   std::cout<<"ref function call"<<std::endl;
}


TEST(Ctor,Vec_Simple){

   Vec v(100);
   test_function_ref(v);
   test_function_val(v);

}


int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

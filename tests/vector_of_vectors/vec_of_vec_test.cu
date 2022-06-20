#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>

#define IS_TRUE(x)\
   { if (!(x)) {\
   std::cout << __FUNCTION__ << "\033[1;31m failed on line \033[0m " << __LINE__<<std::endl;\
   exit(1);\
   }else{\
   std::cout << "\033[1;32m ****OK**** \033[0m\n"<<std::endl; \
   }\
} 


typedef split::SplitVector<int> vec ;
typedef split::SplitVector<double> dvec ;


bool vec_of_vec(){
   split::SplitVector<vec> a(10); 

   a[0]=vec(2,2);
   a[1]=vec(4,4);
   a[2]=vec(8,8);
   a[3]=vec(16,16);
   a[4]=vec(32,32);
   a[5]=vec(64,64);
   a[6]=vec(128,128);
   a[7]=vec(256,256);
   a[8]=vec(512,512);
   a[9]=vec(1024,1024);

   for (int i =0; i<10; i++){
      a[i].print();
   }
   a[0]=vec(2048,2048);
   std::cout<<"***********************"<<std::endl;
   for (int i =0; i<10; i++){
      a[i].print();
   }

   

   return true;
}



void perform_tests(){
   IS_TRUE(vec_of_vec());
    std::cout << "\033[1;32m ==========> All OK <========== \033[0m\n"<<std::endl; \
}


int main(){
   perform_tests();
}

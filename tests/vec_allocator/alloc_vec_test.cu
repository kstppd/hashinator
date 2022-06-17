#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>

typedef split::SplitVector<int> vec ;


bool test(){
   int N=1e8;
   vec a(N);
   std::fill(a.begin(),a.end(),1);
   a.print();
   return true;
}

int main(){
   
   test();

}

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"
using namespace std::chrono;
using type_t = uint32_t;
using  splitvector = split::SplitVector<type_t> ;
constexpr int R = 1000;
#define PROFILE_START(msg)  
#define PROFILE_END() 

template <class Fn, class ... Args>
auto timeMe(Fn fn, Args && ... args){
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   return total_time;
}

template <typename T>
void fillVec(T& vec,size_t sz){
   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_int_distribution<std::mt19937::result_type> dist(0,std::numeric_limits<type_t>::max());
   for (size_t i=0; i< sz;++i){
      vec[i]=i;//dist(rng);
   }
   return;
}

template <typename T>
void printVec(const T& vec){
   for (size_t i=0; i< vec.size();++i){
      std::cout<<vec[i]<<",";
   }
   std::cout<<std::endl;
   return;
}


int main(int argc, char* argv[]){

   int sz=10;
   if (argc>=2){
      sz=atoi(argv[1]);
   }
   size_t N = sz;
   splitvector v0(N),v0_out(N);
   srand(1);
   fillVec(v0,N);
   v0.optimizeGPU();
   v0_out.optimizeGPU();
   SPLIT_CHECK_ERR( split_gpuDeviceSynchronize() );
   for (size_t i =0 ; i < R ; ++i){
      PROFILE_START("Stream_Compaction");
#if 0 
      auto pred =[]__host__ __device__ (type_t  element)->bool{ return (element>10) ;};
      split::tools::copy_if(v0,v0_out,pred);
#else
      auto pred_ =[]__host__ __device__ (type_t  element)->bool{ return (element%2)==0 ;};
      split::tools::copy_if(v0,v0_out,pred_);
#endif
      PROFILE_END();
      break;
   }
   return 0;
}










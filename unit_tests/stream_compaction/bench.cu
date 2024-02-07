#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <nvToolsExt.h>
using namespace std::chrono;
using type_t = uint32_t;
using  splitvector = split::SplitVector<type_t> ;
using  thrustvector = thrust::device_vector<type_t> ;
constexpr int R = 100;
#define PROFILE_START(msg)   nvtxRangePushA((msg))
#define PROFILE_END() nvtxRangePop()

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


void stream_compaction_split(splitvector& v,splitvector& output,  type_t* stack, size_t sz){
   auto pred =[]__host__ __device__ (type_t  element)->bool{ return (element%2)==0 ;};
   split::tools::copy_if(v.data(),output.data(),sz,pred,(void*)stack,sz);
}

void stream_compaction_thrust(thrustvector& v,thrustvector& output){
   auto pred =[]__host__ __device__ (type_t  element)->bool{ return (element%2)==0 ;};
   thrust::copy_if(thrust::device, v.begin(), v.end(), output.begin(), pred);
}

int main(int argc, char* argv[]){


   int sz=10;
   if (argc>=2){
      sz=atoi(argv[1]);
   }
   size_t N = 1<<sz;
   srand(1);
   splitvector v0(N),v0_out(N);
   thrustvector v1(N),v1_out(N);
   fillVec(v0,N);
   fillVec(v1,N);
   splitvector stack(N);
   v0.optimizeGPU();
   v0_out.optimizeGPU();
   stack.optimizeGPU();
   split_gpuDeviceSynchronize();

   
   double t_split={0};
   double t_thrust={0};
   for (int i =0; i<R; i++){

   v0.optimizeGPU();
   v0_out.optimizeGPU();
   stack.optimizeGPU();
      PROFILE_START("THRUST");
      stream_compaction_thrust(v1,v1_out);
      t_thrust+=timeMe(stream_compaction_thrust,v1,v1_out);
      PROFILE_END();

      PROFILE_START("SPLIT");
      stream_compaction_split(v0,v0_out,stack.data(),N);
      t_split+=timeMe(stream_compaction_split,v0,v0_out,stack.data(),N);
      PROFILE_END();
   }
   printf("%d\t%f\t%f\n",sz,t_split/R,t_thrust/R);;
   return 0;
}

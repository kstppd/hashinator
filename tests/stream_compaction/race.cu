#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include "../../src/splitvector/splitvec.h"
#include <cuda_profiler_api.h>
#include "../../src/splitvector/split_tools.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>


typedef uint32_t val_type;
typedef split::SplitVector<val_type,split::split_unified_allocator<val_type>,split::split_unified_allocator<size_t>> split_vector; 

using namespace std::chrono;

struct Predicate{
   __host__ __device__
   bool operator ()(int i)const {
      return i%2==0;
   }
};


// If fn returns void, only the time is returned
template <class Fn, class ... Args>
auto timer(char* name,int reps,Fn fn, Args && ... args){
   static_assert(std::is_void<decltype(fn(args...))>::value,
                "Call timer for non void return type");
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   for(int i =0; i<reps; ++i){
      start = std::chrono::high_resolution_clock::now();
      fn(args...);
      stop = std::chrono::high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop- start).count();
      total_time+=duration;
   }
   std::cout<<name<<" took "<<total_time/reps<<" us | reps= "<<reps<<std::endl;
}
 

void split_test_raw_compaction(size_t size){
   split_vector input_split(size);
   split_vector output_split(size);
   for (size_t i =0 ;  i< size ;++i){
      input_split[i]=i;//tmp;
   }
   input_split.optimizeGPU();
   output_split.optimizeGPU();
   split_tools::copy_if_raw<val_type,Predicate,1024,32>(input_split,output_split,Predicate());
}

void split_test_compaction(size_t size){
   split_vector input_split(size);
   split_vector output_split(size);

   for (size_t i =0 ;  i< size ;++i){
      input_split[i]=i;//tmp;
   }
   input_split.optimizeGPU();
   output_split.optimizeGPU();
   split_tools::copy_if<val_type,Predicate,1024,32>(input_split,output_split,Predicate());
}

void thrust_test_compaction(size_t size){

   thrust::device_vector<val_type> input_thrust(size);
   thrust::device_vector<val_type> output_thrust(size);
   for (size_t i =0 ;  i< size ;++i){
      //auto tmp=dist(rng);
      input_thrust[i]=i;//tmp;
   }

   auto result_end = thrust::copy_if(thrust::device,input_thrust.begin(), input_thrust.end(), output_thrust.begin(), Predicate());
   output_thrust.erase(result_end,output_thrust.end());
}



void split_test_prefix(size_t size){
   split_vector input_split(size);
   split_vector output_split(size);

   for (size_t i =0 ;  i< size ;++i){
      input_split[i]=i;//tmp;
   }
   split_tools::split_prefix_scan<val_type,1024,32>(input_split,output_split);
}
void split_prefix_raw(size_t size){
}
void thrust_test_prefix(size_t size){
   thrust::device_vector<val_type> input_thrust(size);
   thrust::device_vector<val_type> output_thrust(size);
   for (size_t i =0 ;  i< size ;++i){
      input_thrust[i]=i;//tmp;
   }
   thrust::exclusive_scan(thrust::device, input_thrust.begin(),input_thrust.end(), output_thrust.begin(), 0); // in-place scan
}




int main(int argc, char **argv ){


   int power = strtol(argv[1], NULL, 10); 
   uint32_t N=1<<power;
   int reps=100;

   //timer("Split Compaction",reps,split_test_compaction,N);
   //timer("Split Compaction Raw",reps,split_test_raw_compaction,N);
   timer("Split Prefix",reps,split_test_prefix,N);
   timer("Thrust Prefix",reps,thrust_test_prefix,N);
   //timer("Thrust Compaction Raw",reps,thrust_test_compaction,N);

}

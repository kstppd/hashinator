#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include "../../../include/splitvector/splitvec.h"
#include "../../../include/splitvector/split_tools.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>


typedef uint32_t val_type;
typedef split::SplitVector<val_type> split_vector; 

using namespace std::chrono;

struct Predicate{
   __host__ __device__
   bool operator ()(int i)const {
      return i%2==0;
   }
};


// If fn returns void, only the time is returned
template <class Fn, class ... Args>
auto timer(const char* name,int reps,Fn fn, Args && ... args){
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


bool verify(const split_vector& l, const thrust::device_vector<val_type>& r){

   for (size_t i=0; i< l.size(); ++i){
      bool ok = l[i]==r[i];
      if (!ok){return false;}
   }
   return true;
}

template <typename T
         ,typename std::enable_if<std::is_base_of<split_vector, T>::value || 
                   std::is_base_of<thrust::device_vector<val_type>, T>::value >::type* = nullptr>
static inline std::ostream& operator<<(std::ostream& os, T& obj ){
    for (int i=0; i< obj.size();++i){
      os<<obj[i]<<" ";
    }
    return os;
}


void split_test_prefix(split_vector& input_split,split_vector& output_split,size_t size){
   for (size_t i =0 ;  i< size ;++i){
      input_split[i]=i;//tmp;
   }

   //split::tools::splitStackArena mPool(1024*64);


   input_split.optimizeGPU();
   output_split.optimizeGPU();
   split_gpuDeviceSynchronize();
   split::tools::split_prefix_scan(input_split,output_split);


 /*  val_type* in; */
   /*val_type* out; */
   /*split_gpuMalloc( (void**)&in , size*sizeof(val_type));*/
   /*split_gpuMalloc( (void**)&out, size*sizeof(val_type));*/
   /*split_gpuMemcpy(in,input_split.data(),size*sizeof(val_type),split_gpuMemcpyDeviceToDevice);*/
   /*split_gpuMemset(out, 0, size*sizeof(val_type));*/
   /*split::tools::split_prefix_scan_raw<val_type,1024,32>(in,out,mPool,input_split.size());*/
   /*split_gpuMemcpy(output_split.data(),out,size*sizeof(val_type),split_gpuMemcpyDeviceToHost);*/
   /*split_gpuFree(in);*/
      /*split_gpuFree(out);*/
}

void split_test_compaction(split_vector& input_split,split_vector& output_split,size_t size){
   for (size_t i =0 ;  i< size ;++i){
      input_split[i]=i;//tmp;
   }
   split::tools::copy_if(input_split,output_split,Predicate());
}

void thrust_test_prefix(thrust::device_vector<val_type>& input_thrust,thrust::device_vector<val_type>& output_thrust  ,size_t size){
   for (size_t i =0 ;  i< size ;++i){
      input_thrust[i]=i;//tmp;
   }
   thrust::exclusive_scan(thrust::device, input_thrust.begin(),input_thrust.end(), output_thrust.begin(), 0); 
}

void thrust_test_compaction(thrust::device_vector<val_type>& input_thrust,thrust::device_vector<val_type>& output_thrust  ,size_t size){
   for (size_t i =0 ;  i< size ;++i){
      input_thrust[i]=i;//tmp;
   }
   auto res=thrust::copy_if(thrust::device, input_thrust.begin(),input_thrust.end(), output_thrust.begin(),Predicate() ); 
   output_thrust.erase(res,output_thrust.end());
}


bool run_test(int power){

   size_t N=1<<power;
   bool ok= true;
   {
      split_vector input_split(N);
      split_vector output_split(N);
      thrust::device_vector<val_type> input_thrust(N);
      thrust::device_vector<val_type> output_thrust(N);
      int reps=1;

      timer("Split Prefix",reps,split_test_prefix,input_split,output_split,N);
      timer("Thrust Prefix",reps,thrust_test_prefix,input_thrust,output_thrust,N);
      bool ok_scan = verify(output_split,output_thrust);
      ok =ok && ok_scan;
   }

   {
      split_vector input_split(N);
      split_vector output_split(N);
      thrust::device_vector<val_type> input_thrust(N);
      thrust::device_vector<val_type> output_thrust(N);
      int reps=1;
      timer("Split Compaction",reps,split_test_compaction,input_split,output_split,N);
      timer("Thrust Compaction",reps,thrust_test_compaction,input_thrust,output_thrust,N);
      bool ok_comp = verify(output_split,output_thrust);
      ok =ok && ok_comp;
   }
   return ok ;
}

TEST(StremCompaction , Compaction_Tests){
   for (size_t power=2;power<18; ++power){
      bool res = run_test(power);
   }
}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

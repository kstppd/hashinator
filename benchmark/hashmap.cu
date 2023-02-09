#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include "../include/hashinator/hashinator.h"

typedef uint32_t keyval_type;
using namespace Hashinator;
typedef split::SplitVector<keyval_type,split::split_unified_allocator<keyval_type>,split::split_unified_allocator<size_t>> vector ;
using namespace std::chrono;
typedef Hashmap<keyval_type,keyval_type> hashmap;


template <class Fn, class ... Args>
auto execute_and_time(const char* name,Fn fn, Args && ... args) ->bool{
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   bool retval=fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   std::cout<<name<<" took "<<total_time<<" us"<<std::endl;
   return retval;
}

void fill_input(keyval_type* keys , keyval_type* vals, size_t size){
   for (size_t i=0; i<size; ++i){
      keys[i]=i;
      vals[i]=rand()%1000000;
   }
}

bool recover_elements(const hashmap& hmap, keyval_type* keys, keyval_type* vals,size_t size){
   for (size_t i=0; i<size; ++i){
      const cuda::std::pair<keyval_type,keyval_type> kval(keys[i],vals[i]);
      auto retval=hmap.find(kval.first);
      if (retval==hmap.end()){assert(0&& "END FOUND");}
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   }
   return true;
}

bool test_hashmap_insertionDM(keyval_type power,int reps){
   size_t N = 1<<power;
   vector keys(N);
   vector vals(N);
   fill_input(keys.data(),vals.data(),N);
   fill_input(keys.data(),vals.data(),N);
   keys.optimizeGPU();
   vals.optimizeGPU();


   keyval_type* dkeys;
   keyval_type* dvals;
   cudaMalloc(&dkeys, N*sizeof(keyval_type)); 
   cudaMalloc(&dvals, N*sizeof(keyval_type)); 
   cudaMemcpy(dkeys,keys.data(),N*sizeof(keyval_type),cudaMemcpyHostToDevice);
   cudaMemcpy(dvals,vals.data(),N*sizeof(keyval_type),cudaMemcpyHostToDevice);

   hashmap hmap;
   for (int i=0; i<reps;++i){
      hmap.insert(dkeys,dvals,N); 
      std::cout<<hmap.load_factor()<<std::endl;
      hmap.clear();
   }


   
   //Let's also retrieve the keys
   hmap.insert(dkeys,dvals,N); 
   for (int i=0; i<reps;++i){
      hmap.retrieve(dkeys,dvals,N);
   }

   cudaFree(dkeys);
   cudaFree(dvals);
   return true;
}

bool test_hashmap_insertionDM_lf(keyval_type power,int reps,float targetLF){
   //not beautiful
   size_t N = 1<<power;
   double extra=  2*(targetLF-0.5)*N;
   N+=extra;
   std::cout<<"Adding extra keys-> "<<N<<" "<<extra<<std::endl;
   vector keys(N);
   vector vals(N);
   fill_input(keys.data(),vals.data(),N);
   fill_input(keys.data(),vals.data(),N);
   keys.optimizeGPU();
   vals.optimizeGPU();


   keyval_type* dkeys;
   keyval_type* dvals;
   cudaMalloc(&dkeys, N*sizeof(keyval_type)); 
   cudaMalloc(&dvals, N*sizeof(keyval_type)); 
   cudaMemcpy(dkeys,keys.data(),N*sizeof(keyval_type),cudaMemcpyHostToDevice);
   cudaMemcpy(dvals,vals.data(),N*sizeof(keyval_type),cudaMemcpyHostToDevice);

   hashmap hmap;
   for (int i=0; i<reps;++i){
      hmap.insert(dkeys,dvals,N,1); 
      std::cout<<hmap.load_factor()<<" "<<hmap.bucket_count()<<std::endl;
      hmap.clear();
   }
   
   hmap.insert(dkeys,dvals,N,1); 
   for (int i=0; i<reps;++i){
      hmap.retrieve(dkeys,dvals,N); 
   }




   cudaFree(dkeys);
   cudaFree(dvals);
   return true;
}


void driver_insertion(int power){
   int reps=5;
   std::string name= "Power= "+std::to_string(power);
   bool retval = execute_and_time(name.c_str(),test_hashmap_insertionDM ,power,reps);
}

void driver_insertion_lf(int power,float lf){
   int reps=5;
   std::string name= "Power= "+std::to_string(power);
   bool retval = execute_and_time(name.c_str(),test_hashmap_insertionDM_lf ,power,reps,lf);
}

int main(int argc, char* argv[]){
   if (argc<2){return 1;}
   int N =  atoi(argv[1]);
   if (argc==2){
      driver_insertion(N);
   }else if (argc==3){
      driver_insertion_lf(N,atof(argv[2]));
   }
   return 0;
}

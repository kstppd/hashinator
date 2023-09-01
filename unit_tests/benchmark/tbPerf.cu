#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <unordered_set>
#include <random>
#include "../../include/hashinator/hashinator.h"
constexpr int R = 10;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>> vector ;
typedef split::SplitVector<key_type> key_vec;
typedef split::SplitVector<val_type> val_vec;
using hashmap= Hashmap<key_type,val_type>;



auto generateNonDuplicatePairs(vector& src,const size_t size)->void {
    std::unordered_set<int> keys;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<val_type> dist(1, std::numeric_limits<val_type>::max());

    src.clear();
    while (src.size() < size) {
        val_type key = dist(gen);
        // Check if the key is already present
        if (keys.find(key) == keys.end()) {
           val_type val=static_cast<val_type>(key/2);
            src.push_back({key,val});
            keys.insert(key);
        }
    }
}

auto generateNonDuplicatePairs(key_vec &keys,val_vec& vals,const size_t size)->void {
    std::unordered_set<int> unique_keys;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<val_type> dist(1, std::numeric_limits<val_type>::max());
    keys.clear();
    vals.clear();
    while (keys.size() < size) {
        val_type key = dist(gen);
        // Check if the key is already present
        if (unique_keys.find(key) == unique_keys.end()) {
           val_type val=static_cast<val_type>(key/2);
            keys.push_back(key);
            vals.push_back(val);
            unique_keys.insert(key);
        }
    }
}

template <class Fn, class ... Args>
auto timeMe(Fn fn, Args && ... args){
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<milliseconds>(stop- start).count();
   total_time+=duration;
   return total_time;
}

void benchInsert(hashmap& hmap,key_type* gpuKeys, val_type* gpuVals,int sz,float deleteRatio){
   hmap.insert(gpuKeys,gpuVals,1<<sz,1);
   hmap.erase(gpuKeys,0.5*( 1<<sz ));
   hmap.insert(gpuKeys,gpuVals,1<<sz,1);
   //hmap.insert(gpuKeys,gpuVals,1<<sz,1);
   hmap.stats();
   //hmap.erase(gpuKeys,deleteRatio*(1<<sz));
   //hmap.insert(gpuKeys,gpuVals,1<<sz);
   return ;
}

void benchInsert2(hashmap& hmap,vector& src,key_vec& keys, val_vec& vals,int sz,float deleteRatio){
   hmap.insert(src.data(),1<<sz,1);
   hmap.stats();
   hmap.erase(keys.data(),deleteRatio*(1<<sz));
   hmap.stats();
   hmap.insert(src.data(),1<<sz,1);
   hmap.stats();
   return ;
}

int main(int argc, char* argv[]){

   int sz= 14;
   float deleteRatio = 0.5;
   if (argc>=2){
      deleteRatio = atof(argv[1]);
   }
   hashmap hmap(sz+1);
   hmap.optimizeGPU();
   vector cpu_src;
   key_vec cpu_keys;
   val_vec cpu_vals;
   generateNonDuplicatePairs(cpu_keys,cpu_vals,1<<sz);
   generateNonDuplicatePairs(cpu_src,1<<sz);
   cpu_src.optimizeGPU();
   std::cout<<"Generated "<<cpu_keys.size()<<" unique keys!"<<std::endl;

   key_type* gpuKeys;
   val_type* gpuVals;
   split_gpuMalloc((void **) &gpuKeys, (1<<sz)*sizeof(key_type));
   split_gpuMalloc((void **) &gpuVals, (1<<sz)*sizeof(val_type));
   split_gpuMemcpy(gpuKeys,cpu_keys.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice);
   split_gpuMemcpy(gpuVals,cpu_vals.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice);

   key_vec keyBuffer;
   val_vec valBuffer;
   for (auto& i : cpu_src){
      keyBuffer.push_back(i.first);
      valBuffer.push_back(i.first);
   }
   keyBuffer.optimizeGPU();
   valBuffer.optimizeGPU();

   double t={0};
   for (int i =0; i<R; i++){
      hmap.optimizeGPU();
      keyBuffer.optimizeGPU();
      valBuffer.optimizeGPU();
      cpu_src.optimizeGPU();
      //t+=timeMe(benchInsert,hmap,gpuKeys,gpuVals,sz,deleteRatio);
       t+= timeMe(benchInsert2,hmap,cpu_src,keyBuffer,valBuffer,sz,deleteRatio);
      hmap.clear();
   }
   std::cout<<"Done in "<<t/R<<" ms"<<std::endl;

   split_gpuFree(gpuKeys);
   split_gpuFree(gpuVals);
   return 0;

}

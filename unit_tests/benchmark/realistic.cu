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

void benchInsert2(hashmap& hmap, hash_pair<key_type,val_type>*src,key_type* keys, val_type* vals,int sz,float deleteRatio){
   hmap.insert(src,1<<sz,1);
   hmap.retrieve(keys,vals,1<<sz);
   hmap.erase(keys,1<<sz);
   //hmap.clean_tombstones();
   hmap.insert(src,1<<sz,1);
   hmap.retrieve(keys,vals,1<<sz);
   (void)deleteRatio;
   return ;
}


int main(int argc, char* argv[]){

   int sz= 10;
   if (argc>=2){
      sz=atoi(argv[1]);
   }
   float deleteRatio = 1.0;
   hashmap hmap(sz+1);
   hmap.optimizeGPU();
   vector cpu_src;
   key_vec cpu_keys;
   val_vec cpu_vals;
   generateNonDuplicatePairs(cpu_src,1<<sz);
   for (auto i: cpu_src){
      cpu_keys.push_back(i.first);
      cpu_vals.push_back(i.second);
   }
   cpu_src.optimizeGPU();
   std::cout<<"Generated "<<cpu_keys.size()<<" unique keys!"<<std::endl;
   key_type* gpuKeys;
   val_type* gpuVals;
   hash_pair<key_type,val_type>* gpuPairs;
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &gpuKeys, (1<<sz)*sizeof(key_type)) );
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &gpuVals, (1<<sz)*sizeof(val_type)) );
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &gpuPairs, (1<<sz)*sizeof(hash_pair<key_type,val_type>)) );
   SPLIT_CHECK_ERR( split_gpuMemcpy(gpuKeys,cpu_keys.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice) );
   SPLIT_CHECK_ERR( split_gpuMemcpy(gpuVals,cpu_vals.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice) );
   SPLIT_CHECK_ERR( split_gpuMemcpy(gpuPairs,cpu_src.data(),(1<<sz)*sizeof(hash_pair<key_type,val_type>),split_gpuMemcpyHostToDevice) );
   int device;
   cudaGetDevice(&device);
   hmap.memAdvise(cudaMemAdviseSetPreferredLocation,device);
   hmap.memAdvise(cudaMemAdviseSetAccessedBy,device);
   cudaDeviceSynchronize();


   double t={0};
   for (int i =0; i<R; i++){
      hmap.optimizeGPU();
      t+=timeMe(benchInsert2,hmap,gpuPairs,gpuKeys,gpuVals,sz,deleteRatio);
      hmap.clear(targets::host);
   }
   std::cout<<"Done in "<<t/R<<" us"<<std::endl;

   return 0;

}

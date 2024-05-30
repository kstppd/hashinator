#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <unordered_set>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include <nvToolsExt.h>
#define PROFILE_START(msg)   nvtxRangePushA((msg))
#define PROFILE_END() nvtxRangePop()
constexpr int R = 50;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>> vector ;
typedef split::SplitVector<key_type> key_vec;
typedef split::SplitVector<val_type> val_vec;
using hashmap= Hashmap<key_type,val_type>;


static void *stack=nullptr;
static size_t bytes = 1024*1024;

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
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   return total_time;
}

void benchInsert(hashmap& hmap,key_type* gpuKeys, val_type* gpuVals,int sz){
   hmap.optimizeGPU();
   hmap.insert(gpuKeys,gpuVals,1<<sz,1);
   return ;
}

void benchRetrieve(hashmap& hmap,key_type* gpuKeys, val_type* gpuVals,int sz){
   hmap.optimizeGPU();
   hmap.retrieve(gpuKeys,gpuVals,1<<sz);
   return ;
}

void benchExtract(hashmap& hmap,key_vec& spare,int sz){
   ( void )sz;
   spare.optimizeGPU();
   hmap.optimizeGPU();
   //hmap.extractAllKeys(spare,stack,bytes);
   hmap.extractAllKeys(spare);
   return ;
}

void benchErase(hashmap& hmap,key_type* gpuKeys, val_type* gpuVals,int sz){
   (void)gpuVals;
   hmap.optimizeGPU();
   hmap.erase(gpuKeys,1<<sz);
   return ;
}

int main(int argc, char* argv[]){

   int sz= 18;
   if (argc>=2){
      sz=atoi(argv[1]);
   }
   hashmap hmap(sz+1);
   int device;
   SPLIT_CHECK_ERR( split_gpuGetDevice(&device) );
   hmap.memAdvise(cudaMemAdviseSetPreferredLocation,device);
   hmap.memAdvise(cudaMemAdviseSetAccessedBy,device);
   hmap.optimizeGPU();
   hmap.optimizeGPU();
   double t_insert={0};
   double t_retrieve={0};
   double t_extract={0};
   double t_erase={0};

   for (int i =0; i<R; i++){
   vector cpu_src;
   key_vec cpu_keys;
   val_vec cpu_vals;
   generateNonDuplicatePairs(cpu_keys,cpu_vals,1<<sz);
   //std::cout<<"Generated "<<cpu_keys.size()<<" unique keys!"<<std::endl;


   key_vec  spare;
   spare.resize(1<<sz);
   key_type* gpuKeys;
   val_type* gpuVals;
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &gpuKeys, (1<<sz)*sizeof(key_type)) );
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &stack, bytes) );
   SPLIT_CHECK_ERR( split_gpuMalloc((void **) &gpuVals, (1<<sz)*sizeof(val_type)) );
   SPLIT_CHECK_ERR( split_gpuMemcpy(gpuKeys,cpu_keys.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice) );
   SPLIT_CHECK_ERR( split_gpuMemcpy(gpuVals,cpu_vals.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice) );
      hmap.optimizeGPU();

      PROFILE_START("insert");
      t_insert+=timeMe(benchInsert,hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
   
      PROFILE_START("extract");
      t_extract+=timeMe(benchExtract,hmap,spare,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
    
      PROFILE_START("retrieve");
      t_retrieve+=timeMe(benchRetrieve,hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
      
      PROFILE_START("erase");
      t_erase+=timeMe(benchErase,hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
      
      hmap.clear();
   SPLIT_CHECK_ERR( split_gpuFree(gpuKeys) );
   SPLIT_CHECK_ERR( split_gpuFree(gpuVals) );
   SPLIT_CHECK_ERR( split_gpuFree(stack) );
   }
   t_insert/=(float)R;
   t_retrieve/=(float)R;
   t_extract/=(float)R;
   t_erase/=(float)R;

   printf("%d %d %d %d %d\n",sz,(int)t_insert,(int)t_retrieve,(int)t_extract,(int)t_erase);
   return 0;

}

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <unordered_set>
#include <random>
constexpr int R = 10;
#include "../../include/hashinator/hashmap/hashmap.h"
#include <nvToolsExt.h>
#define PROFILE_START(msg)   nvtxRangePushA((msg))
#define PROFILE_END() nvtxRangePop()

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


__global__
void gpu_recover_warpWide(hashmap* hmap,key_type*keys ,val_type* vals,size_t N  ){

   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t wid = index / Hashinator::defaults::WARPSIZE;
   const size_t w_tid = index % defaults::WARPSIZE;
   if (wid < N ){
      key_type key=keys[wid];
      hmap->warpFind(key,vals[wid],w_tid);
   }
}

void benchRetrieveWarpWide(hashmap* hmap,key_type* gpuKeys, val_type* gpuVals,int sz){
   constexpr size_t warpsize =  Hashinator::defaults::WARPSIZE;
   size_t threadsNeeded =(1<<sz)*warpsize; 
   constexpr size_t blocksize=1024;
   size_t blocks = threadsNeeded/blocksize;
   blocks+=(blocks==0);
   gpu_recover_warpWide<<<blocks,blocksize>>>(hmap,gpuKeys,gpuVals,sz);
   split_gpuDeviceSynchronize();
}

int main(int argc, char* argv[]){

   int sz= 24;
   if (argc>=2){
      sz=atoi(argv[1]);
   }
   hashmap*hmap=new hashmap(sz+1);
   int device;
   split_gpuGetDevice(&device);
   hmap->optimizeGPU();
   hmap->optimizeGPU();
   vector cpu_src;
   key_vec cpu_keys;
   val_vec cpu_vals;
   generateNonDuplicatePairs(cpu_keys,cpu_vals,1<<sz);
   //std::cout<<"Generated "<<cpu_keys.size()<<" unique keys!"<<std::endl;

   key_type* gpuKeys;
   key_vec  spare;
   val_type* gpuVals;
   split_gpuMalloc((void **) &gpuKeys, (1<<sz)*sizeof(key_type));
   split_gpuMalloc((void **) &stack, bytes);
   split_gpuMalloc((void **) &gpuVals, (1<<sz)*sizeof(val_type));
   split_gpuMemcpy(gpuKeys,cpu_keys.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice);
   split_gpuMemcpy(gpuVals,cpu_vals.data(),(1<<sz)*sizeof(key_type),split_gpuMemcpyHostToDevice);
   spare.resize(1<<sz);

   double t_insert={0};
   double t_retrieveWarpWide={0};
   double t_retrieve={0};
   double t_extract={0};
   double t_erase={0};

   for (int i =0; i<R; i++){
      hmap->optimizeGPU();

      PROFILE_START("insert");
      t_insert+=timeMe(benchInsert,*hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
   
      PROFILE_START("extract");
      t_extract+=timeMe(benchExtract,*hmap,spare,sz);
      cudaDeviceSynchronize();
      PROFILE_END();

      PROFILE_START("retrieve");
      t_retrieve+=timeMe(benchRetrieve,*hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();

      PROFILE_START("retrieveWarpWide");
      t_retrieveWarpWide+=timeMe(benchRetrieveWarpWide,hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
      
      PROFILE_START("erase");
      t_erase+=timeMe(benchErase,*hmap,gpuKeys,gpuVals,sz);
      cudaDeviceSynchronize();
      PROFILE_END();
      
      hmap->clear();
   }
   t_insert/=(float)R;
   t_retrieve/=(float)R;
   t_extract/=(float)R;
   t_erase/=(float)R;

   printf("%d %d %d %d %d %d\n",sz,(int)t_insert,(int)t_retrieve,(int)t_retrieveWarpWide,(int)t_extract,(int)t_erase);
   split_gpuFree(gpuKeys);
   split_gpuFree(gpuVals);
   split_gpuFree(stack);
   return 0;

}

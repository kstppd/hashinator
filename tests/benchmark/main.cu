#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
constexpr int N = 24;
constexpr int R = 10;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
using hashmap= Hashmap<key_type,val_type>;


void create_input(hash_pair<key_type,val_type>* src, int sz){
   for (size_t i=0; i<(1<<sz); ++i){
      src[i].first=i;
      src[i].second=i;
   }
}

void benchInsert(hashmap& hmap,hash_pair<key_type,val_type>* src,int sz){
   hmap.insert(src,1<<sz);
   assert(hmap.peek_status()==status::success);
   hmap.stats();
   hmap.clear();
   return ;
}

int main(int argc, char* argv[]){

   const int sz=N;
   hashmap hmap(sz+1);
   hmap.optimizeGPU();
   vector cpu_src(1<<sz);
   create_input(cpu_src.data(),sz);
   hash_pair<key_type,val_type>* src;
   cudaMalloc((void **) &src, (1<<sz)*sizeof(hash_pair<key_type,val_type>));
   cudaMemcpy(src,cpu_src.data(),cpu_src.size()*sizeof(hash_pair<key_type,val_type>),cudaMemcpyHostToDevice);

   for (int i =0; i<R; i++){
      hmap.optimizeGPU();
      benchInsert(hmap,src,sz);
   }

   cudaFree(src);
   return 0;

}

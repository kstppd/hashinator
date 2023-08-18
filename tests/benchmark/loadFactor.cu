#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
constexpr int R = 10;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
using hashmap= Hashmap<key_type,val_type>;


void create_input(hash_pair<key_type,val_type>* src, size_t N){
   for (size_t i=0; i<N; ++i){
      src[i].first=i;
      src[i].second=i;
   }
}

void benchInsert(hashmap& hmap,hash_pair<key_type,val_type>* src,vector& buffer ,size_t N){
   hmap.insert(src,N,1);
   hmap.retrieve(buffer.data(),N);
   assert(hmap.peek_status()==status::success);
   hmap.stats();
   hmap.clear();
   return ;
}

int main(int argc, char* argv[]){

   const int sz=24;
   float targetLF=0.5;
   if (argc>2){
      targetLF=atof(argv[1]);
   }
   const size_t N = (1<<(sz+1))*targetLF;
   std::cout<<targetLF<< " "<<N<<std::endl;
   hashmap hmap(sz+1);
   hmap.optimizeGPU();
   vector cpu_src(N);
   create_input(cpu_src.data(),N);
   hash_pair<key_type,val_type>* src;
   cudaMalloc((void **) &src, N*sizeof(hash_pair<key_type,val_type>));
   cudaMemcpy(src,cpu_src.data(),cpu_src.size()*sizeof(hash_pair<key_type,val_type>),cudaMemcpyHostToDevice);

   for (int i =0; i<R; i++){
      hmap.optimizeGPU();
      cpu_src.optimizeGPU();
      benchInsert(hmap,src,cpu_src,N);
   }

   cudaFree(src);
   return 0;

}

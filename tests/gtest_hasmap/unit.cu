#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"

typedef uint32_t val_type;

__global__
void fillMap(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
    dmap->set_element(index, index);
}


void load_to_50_pc(Hashinator<val_type,val_type>& map ,int threads){
   //timed block
   size_t total_keys=map.bucket_count()/2;
   size_t total_threads=threads;
   size_t total_blocks= total_keys/total_threads;
   Hashinator<val_type,val_type>* dmap = map.upload();
   printf("Running with %i Threads and %i Blocks\n ",(int)threads,(int)total_blocks);
   auto start = std::chrono::high_resolution_clock::now();
   fillMap<<<total_blocks,total_threads>>>(dmap);
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   map.clean_up_after_device(dmap);
   map.print_all();
   map.print_kvals();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   //printf("TIME: %.3f seconds for %zu elements at a load factor of %f\n", total_time.count() * 1e-9,map.size(),map.load_factor());

}
 


void test1(){
   Hashinator<val_type,val_type> map;
   map.resize(6);
   load_to_50_pc(map,2);

}



int main(){
   test1();
}


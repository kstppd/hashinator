#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#define N  1<<27

typedef uint32_t val_type;

__global__
void fillMap(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
    dmap->set_element(index, index);
}


__global__
void fillMap_neg(Hashinator<val_type,val_type> *dmap){
   int index = (blockIdx.x * blockDim.x + threadIdx.x)+268435456;
    dmap->set_element(index, index);
}

void load_to_50_pc(Hashinator<val_type,val_type>& map ,int threads){
   //timed block
   map.resize(28);
   size_t total_keys=N;
   size_t total_threads=threads;
   size_t total_blocks= total_keys/total_threads;
   Hashinator<val_type,val_type>* dmap = map.upload();
   //printf("Running with %i Threads and %i Blocks\n ",(int)threads,(int)total_blocks);
   auto start = std::chrono::high_resolution_clock::now();
   fillMap<<<total_blocks,total_threads>>>(dmap);
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   map.clean_up_after_device(dmap);
   map.print_all();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   //printf("TIME: %.3f seconds for %zu elements at a load factor of %f\n", total_time.count() * 1e-9,map.size(),map.load_factor());

}

void load_to_lf(Hashinator<val_type,val_type>& map ,float target_lf){

   map[0]=0;
   float current_lf=map.load_factor();
   size_t size=map.size();
   size_t threads=32;

   int rem_elements= (target_lf*size/current_lf) -size;
   while(rem_elements%threads!=0){
      rem_elements++;
   }
   size_t blocks=rem_elements/threads;
   
   //printf("Running with %i Threads and %i Blocks\n ",(int)threads,(int)blocks);
   Hashinator<val_type,val_type>* dmap = map.upload();
   auto start = std::chrono::high_resolution_clock::now();
   fillMap<<<blocks,threads>>>(dmap);
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   map.clean_up_after_device(dmap);
   map.print_all();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   //printf("TIME: %.3f seconds for %zu elements at a load factor of %f\n", total_time.count() * 1e-9,map.size(),map.load_factor());
   

}

void addNelems(Hashinator<val_type,val_type>&map,int numel){

   int threads=32;
   auto initial_lf=map.load_factor();
   int blocks=numel/threads;
   //printf("Adding %i elements  with %i Threads and %i Blocks\n ",(int)numel,(int)threads,(int)blocks);
   Hashinator<val_type,val_type>* dmap = map.upload();
   auto start = std::chrono::high_resolution_clock::now();
   fillMap_neg<<<blocks,threads>>>(dmap);
   cudaDeviceSynchronize();
   auto end = std::chrono::high_resolution_clock::now();
   map.clean_up_after_device(dmap);
   //map.print_all();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   printf("--->TIME: %.3f seconds for %zu elements at a load factor of %f\n", total_time.count() * 1e-9,map.size(),initial_lf);


}
   
   



int main(){
   Hashinator<val_type,val_type> map;
   map.resize(28);

   int numel=1<<24;
   for (int i=70; i<=70;i+=1){
      load_to_lf(map,i/100.0);
      addNelems(map,numel);
      map.clear();

   }
}


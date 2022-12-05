#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#define N 1<<12

typedef uint32_t val_type;

void cpu_write_map(Hashinator<val_type,val_type>& map,int total_keys=N){
   for (val_type i=0; i<total_keys;i++){
      map[i]=0;
   }
}

__global__
void gpu_write_map(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N){
      //hash_pair<val_type,val_type> p{index,index};
      dmap->set_element(index,index);
      //auto ret=dmap->insert(p);
   }
   return;
}


__global__
void gpu_delete_all(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->find(index);
      dmap->erase(kpos);
   }
   return;
}


__global__
void gpu_delete_even(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->find(index);
      if (kpos==dmap->end()){return;}
      if (kpos->second %2==0 ){
         dmap->erase(kpos);
      }
   }
   return;
}


__global__
void gpu_delete_odd(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->find(index);
      if (kpos==dmap->end()){return;}
      if (kpos->second %2==1){
         dmap->erase(kpos);
      }
   }
   return;
}


void map_test(int power,int threads){

   //We create an instance of hashinator and add elements to it on host
   Hashinator<val_type,val_type> hmap;
   hmap.resize(power);

   //Some magic numbers!( used to launch the kernels)
   //Declare a pointer for use in kernels
   size_t blocks=(1<<power)/threads;
   Hashinator<val_type,val_type>* dmap;
   dmap=hmap.upload();
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   hmap.download();

   dmap=hmap.upload();
   gpu_delete_even<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   hmap.download();



   hmap.print_kvals();
}

int main(int argc, char**argv){
   map_test(5,32);
}

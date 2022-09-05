#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#define N 1<<17

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
      dmap->set_element(index,index);
   }
   return;
}


__global__
void gpu_delete_all(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->d_find(index);
      dmap->d_erase(kpos);
   }
   return;
}



__global__
void gpu_delete_even(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->d_find(index);
      if (kpos==dmap->d_end()){return;}
      if (kpos->second %2==0 || kpos->second==0){
         dmap->d_erase(kpos);
      }
   }
   return;
}


__global__
void gpu_delete_odd(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      auto kpos=dmap->d_find(index);
      if (kpos==dmap->d_end()){return;}
      if (kpos->second %2==1){
         dmap->d_erase(kpos);
      }
   }
   return;
}


int main(int argc, char* argv[]){
   Hashinator<val_type,val_type> hmap;
   cpu_write_map(hmap,N);
   size_t threads=32;
   size_t blocks=1<<26;

   Hashinator<val_type,val_type>* dmap;
   dmap=hmap.upload();
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap);

   dmap=hmap.upload();
   gpu_delete_even<<<blocks,threads>>> (dmap);
   //gpu_delete_even<<<N,1>>> (dmap);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap);

   dmap=hmap.upload();
   gpu_delete_odd<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap);



   return 0;
}


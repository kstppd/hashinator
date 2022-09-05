#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#define N  32

typedef uint32_t val_type;

__global__
void gpu_write_map(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N){
      dmap->set_element(index,index);
   }
   return;
}

__global__
void gpu_delete_even(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N && index%2==0){
      dmap->d_erase(index);
   }
   return;
}



__global__
void gpu_delete_all(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N ){
      dmap->d_erase(index);
   }
   return;
}


__global__
void gpu_delete_odd(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<N && index%2==1){
      dmap->d_erase(index);
   }
   return;
}

void cpu_write_map(Hashinator<val_type,val_type>& map,int total_keys=N){
   for (val_type i=0; i<total_keys;i++){
      map[i]=0;
   }
}




int main(int argc, char* argv[]){
   Hashinator<val_type,val_type> hmap;
   cpu_write_map(hmap,N);
   size_t threads=32;
   size_t blocks=1024;

   Hashinator<val_type,val_type>* dmap=hmap.upload();
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap);


   Hashinator<val_type,val_type>* dmap2=hmap.upload();
   gpu_delete_even<<<blocks,threads>>> (dmap2);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap2);

   Hashinator<val_type,val_type>* dmap3=hmap.upload();
   gpu_delete_odd<<<blocks,threads>>> (dmap3);
   cudaDeviceSynchronize();
   hmap.clean_up_after_device(dmap3);



   return 0;
}


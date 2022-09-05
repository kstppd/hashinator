#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
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
      if (kpos->second %2==0 ){
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

   //We create an instance of hashinator and add elements to it on host
   Hashinator<val_type,val_type> hmap;
   cpu_write_map(hmap,N);

   //Some magic numbers!( used to launch the kernels)
   size_t threads=32;
   size_t blocks=1<<10;

   //Declare a pointer for use in kernels
   Hashinator<val_type,val_type>* dmap;

   //Upload map to device
   dmap=hmap.upload();

   //Call a simple kernel that just writes to the map elements based on their index
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   
   //Always clean up after kernel
   hmap.clean_up_after_device(dmap);

   //Let's reupload the map
   dmap=hmap.upload();

   //Now we delete all even elements
   gpu_delete_even<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();

   //And we clean up again
   hmap.clean_up_after_device(dmap);

   //One more time
   dmap=hmap.upload();
   //And we remove the odd numbers
   gpu_delete_odd<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();

   //We clean up
   hmap.clean_up_after_device(dmap);

   //We now expect the map to have 0 fill as we deleted all the elemets
   std::cout<<"Map should have 0 fill:\n";
   std::cout<<"Map's fill is -->"<<hmap.size()<<std::endl;
   assert(hmap.size()==0 && "Map should have zero 0 but that is not the case!");




   return 0;
}


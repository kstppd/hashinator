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
      std::pair<val_type,val_type> p{index,index};
      //dmap->set_element(index,index);
      auto ret=dmap->insert(p);
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


void stress_test(int power,int threads){

   //We create an instance of hashinator and add elements to it on host
   Hashinator<val_type,val_type> hmap;
   hmap.resize(power);

   //Some magic numbers!( used to launch the kernels)
   size_t blocks=(1<<power)/threads;

   //Declare a pointer for use in kernels
   Hashinator<val_type,val_type>* dmap;

   //Upload map to device
   dmap=hmap.upload();

   //Call a simple kernel that just writes to the map elements based on their index
   gpu_write_map<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();
   
   //Always clean up after kernel
   hmap.download();

   //Let's reupload the map
   dmap=hmap.upload();

   //Now we delete all even elements
   gpu_delete_even<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();

   //And we clean up again
   hmap.download();

   //One more time
   dmap=hmap.upload();

   //And we remove the odd numbers
   gpu_delete_odd<<<blocks,threads>>> (dmap);
   cudaDeviceSynchronize();

   //We clean up
   hmap.download();

   hmap.print_kvals();
   //We now expect the map to have 0 fill as we deleted all the elemets
   assert(hmap.size()==0 && "Map fill should be zero but is not. Something is broken!");
}



int main(int argc, char**argv){
  
   if (argc!=2){
      std::cerr<<"Plase provide maximum number of max element power for this test!.."<<std::endl;
      std::cerr<<"Usage.."<<std::endl;
      std::cerr<<"\t "<<argv[0]<<"<max element power>(11-30)"<<std::endl;
      exit(1);
   }
   int maxElements=atoi(argv[1]);
   if (maxElements<11){
      std::cerr<<"please enter something larger than 11 this is a simple test!"<<std::endl;
      exit(1);
   }
   int threads=8;
   for (int power=5; power<maxElements; power++){
      auto start = std::chrono::high_resolution_clock::now();
      stress_test(power,threads);
      auto end = std::chrono::high_resolution_clock::now();
      auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      printf("TIME: %.5f Power: %d \n", total_time.count() * 1e-9,power);
   }
   return 0;
}

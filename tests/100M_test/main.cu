#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>

typedef uint32_t val_type;


void cpu_write_map(Hashinator<val_type,val_type>& map,size_t total_keys){
   for (val_type i=0; i<total_keys;i++){
      map[i]=0;
   }
}

void cpu_delete_all(Hashinator<val_type,val_type>& map,int total_keys){
   for (val_type i=0; i<total_keys;i++){
      map.erase(i);
   }
}
__global__
void gpu_write_map(Hashinator<val_type,val_type> *dmap,size_t total_keys){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<total_keys){
      dmap->set_element(index,index);
   }
   return;
}


__global__
void gpu_delete_all(Hashinator<val_type,val_type> *dmap,size_t total_keys){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index< total_keys){
      //dmap->d_erase(index);
      dmap->purge(index);
   }
   return;
}



void stress_test_CPU(size_t Nelements){
   Hashinator<val_type,val_type> hmap;
   cpu_write_map(hmap,Nelements);
   hmap.print_all();
   cpu_delete_all(hmap,Nelements);
   //cpu_write_map(hmap,Nelements);
   //hmap.print_all();
   //cpu_delete_all(hmap,Nelements);
}


void stress_test_GPU(size_t Nelements,int threads){

   //We create an instance of hashinator and add elements to it on host
   Hashinator<val_type,val_type> hmap;
   int power=1+log2(Nelements);
   hmap.resize(power);

   //Some magic numbers!( used to launch the kernels)
   size_t blocks=1+(1<<power)/threads;

   //Declare a pointer for use in kernels
   Hashinator<val_type,val_type>* dmap;

   for (int i =0 ; i< 1 ; i++){
      //Upload map to device
      dmap=hmap.upload();

      ////Call a simple kernel that just writes to the map elements based on their index
      gpu_write_map<<<blocks,threads>>> (dmap,Nelements);
      cudaDeviceSynchronize();
      
      //Always clean up after kernel
      hmap.download();
      hmap.print_all();

      //Let's reupload the map
      dmap=hmap.upload();

      //Now we delete all even elements
      gpu_delete_all<<<blocks,threads>>> (dmap,Nelements);
      cudaDeviceSynchronize();

      //And we clean up again
      hmap.download();
      hmap.print_all();
   }

   ////We now expect the map to have 0 fill as we deleted all the elemets
   //assert(hmap.size()==0 && "Map fill should be zero but is not. Something is broken!");
}



int main(int argc, char**argv){
  
   if (argc!=2){
      std::cerr<<"Plase provide the number of total elements for the test.."<<std::endl;
      std::cerr<<"Usage.."<<std::endl;
      std::cerr<<"\t "<<argv[0]<<"<max element power>(11-30)"<<std::endl;
      exit(1);
   }
   size_t maxElements=atoi(argv[1]);
   if (maxElements<1){
      std::cerr<<"please enter something larger than 1 this is a simple test!"<<std::endl;
      exit(1);
   }


   auto start = std::chrono::high_resolution_clock::now();
   stress_test_CPU(maxElements);
   auto end = std::chrono::high_resolution_clock::now();
   auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   printf("CPU time: %.3f seconds for %zu elements\n", total_time.count() * 1e-9,maxElements);


   int threads=32;
   start = std::chrono::high_resolution_clock::now();
   stress_test_GPU(maxElements,threads);
   end = std::chrono::high_resolution_clock::now();
   total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   printf("GPU time: %.3f seconds for %zu elements\n", total_time.count() * 1e-9,maxElements);



}

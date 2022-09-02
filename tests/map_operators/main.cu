#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator/hashinator.h"
#include <gtest/gtest.h>
#define N  64

typedef uint32_t val_type;

__global__
void gpu_write_map(Hashinator<val_type,val_type> *dmap){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   dmap->set_element(index,index);
   return;
}


void cpu_write_map(Hashinator<val_type,val_type>& map,int total_keys=N){
   for (val_type i=0; i<total_keys;i++){
      map[i]=i;
   }
}




int main(int argc, char* argv[]){
   return 0;
}


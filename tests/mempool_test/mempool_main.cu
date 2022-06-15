#include<iostream>
#include<stdlib.h>
#include"../../src/cuda_mempool/mempool.h"

// __global__
// void kernel(UnifiedMemPool* dpool){
//     dpool->malloc(100);
// }


// int main(){
//     UnifiedMemPool p(1<<10);
//     p.malloc(100);
//     p.addBlock(1<<20);
//     kernel<<<1,1>>>(p.upload());
//     cudaDeviceSynchronize();
//     p.stats();

// }



int main(){

   UnifiedMemPool pool;

   void * ptr = pool.allocate(1<<10);
   void * ptr2 = pool.allocate(1<<10);
   pool.stats();
}

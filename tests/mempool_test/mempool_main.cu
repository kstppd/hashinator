#include<iostream>
#include<stdlib.h>
#include"../../src/cuda_mempool/mempool.h"

void check_ptr(void*ptr){

   if (ptr == nullptr){
      abort();
   }
}


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



typedef  DoublyStackedArena DSA;
typedef FixedList<DSA>::Node Node;
int main(){

   UnifiedMemPool pool;
   int* ptr = (int*)pool.allocate(10*sizeof(int));
   for (int i=0; i<10; i++){
      ptr[i]=i;
   }
   
   int* ptr2 = (int*)pool.allocate(20*sizeof(int));
   for (int i=0; i<20; i++){
      ptr2[i]=10+i;
   }



   int* ptr3 = (int*)pool.allocate(10*sizeof(int));
   for (int i=0; i<10; i++){
      ptr3[i]=-i;
   }


   check_ptr(ptr);
   pool.stats(sizeof(int));



   pool.free(ptr2);
   pool.stats(sizeof(int));



   pool.free(ptr);
   pool.stats(sizeof(int));











}

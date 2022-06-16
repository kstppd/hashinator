#include<iostream>
#include<stdlib.h>
#include"../../src/cuda_mempool/mempool.h"

void check_ptr(void*ptr){

   if (ptr == nullptr){
      abort();
   }
}



typedef  DoublyStackedArena DSA;
typedef FixedList<DSA>::Node Node;
int main(){


   {

      UnifiedMemPool pool;
      int* ptr = (int*)pool.allocate(10*sizeof(int));
      for (int i=0; i<10; i++){
         ptr[i]=i;
      }
      
      int* ptr2 = (int*)pool.allocate(20*sizeof(int));
      for (int i=0; i<20; i++){
         ptr2[i]=10+i;
      }



      int* ptr3 = (int*)pool.allocate(100*sizeof(int));
      for (int i=0; i<100; i++){
         ptr3[i]=i;
      }

      int* ptr4 = (int*)pool.allocate(200*sizeof(int));
      for (int i=0; i<200; i++){
         ptr4[i]=100+i;
      }

      check_ptr(ptr);
      pool.stats(sizeof(int));


      pool.free(ptr);
      pool.stats(sizeof(int));


      int* ptr5 = (int*)pool.allocate(10*sizeof(int));
      for (int i=0; i<10; i++){
         ptr5[i]=-i;
      }
      pool.stats(sizeof(int));

   }
}

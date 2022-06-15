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



   typedef  DoublyStackedArena DSA;
   typedef FixedList<DSA>::Node Node;
int main(){


   //nightly example to show proper usage. 
   FixedList<DSA> nodelist;
   Node* block=nodelist.head;
   DSA newStack=DSA(1024);
   block->data=&newStack;
   block->data->allocate(100);
   std::cout<<block<<"  "<<block->data<<" "<<block->data->Size()<<std::endl;
 
   block=block->next;

   DSA newStack_2=DSA(4096);
   block->data=&newStack_2;
   std::cout<<block<<"  "<<block->data<<" "<<block->data->Size()<<std::endl;
   
   

   



   //UnifiedMemPool pool;

   //void * ptr = pool.allocate(1<<10);
   //void * ptr2 = pool.allocate(1<<10);
   //pool.stats();
}

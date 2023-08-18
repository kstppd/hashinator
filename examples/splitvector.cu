#include <iostream>
#include "../include/splitvector/splitvec.h"
#include <iostream>

typedef int val_type;
typedef split::SplitVector<val_type> vector;


static inline std::ostream& operator<<(std::ostream& os, vector& vec ){
   for (const auto&i : vec){
      std::cout<<i<<" ";
   }
   std::cout<<std::endl;
    return os;
}

__global__
void push_back_kernel(vector* a){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   a->device_push_back(index);
}

void basic_host_usage()
{
   vector vec {1,2,3,4,5};
   vec.push_back(4);
   std::cout<<vec<<std::endl;
   vec.clear();
}

void basic_device_usage()
{

   vector vec {1,2,3,4,5};
   vec.reserve(128);
   auto d_vec= vec.upload();
   push_back_kernel<<<1,64>>>(d_vec);
   cudaFree(d_vec);
   std::cout<<vec<<std::endl;
}

int main()
{
   basic_host_usage();
   basic_device_usage();
   return 0;
}

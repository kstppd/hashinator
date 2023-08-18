#include "../include/hashinator/hashinator.h"
#include <iostream>
#include <random>

using namespace Hashinator;
typedef uint32_t val_type;
typedef split::SplitVector<hash_pair<val_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;

__global__ 
void gpu_write(Hashmap<val_type,val_type>* hmap, hash_pair<val_type,val_type>*src, size_t N)
{
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N ){
      hmap->set_element(src[index].first, src[index].second);
   }
}

__global__ 
void gpu_read_and_delete(Hashmap<val_type,val_type>* hmap){
   val_type index = blockIdx.x * blockDim.x + threadIdx.x;
   auto kval= hmap->device_find(index);
   if (kval!=hmap->device_end()){
      hmap->device_erase(kval);
   }
}

void basic_host_usage()
{
   std::cout<<"\nHost Usage\n"<<std::endl;
   Hashmap<val_type,val_type> hmap;

   //Write
   for (uint32_t i=0 ; i<64; ++i){
      hmap[i]=rand()%10000;
   }

   //Read
   for (const auto& i:hmap){
      std::cout<<"["<<i.first<<" "<<i.second<<"] ";
   }
   std::cout<<std::endl;
}

void basic_device_usage()
{
   std::cout<<"\nDevice Usage\n"<<std::endl;
   vector src(64);
   Hashmap<val_type,val_type> hmap;
   hmap.resize(7);
   //Create Input
   for (uint32_t i=0 ; i<64; ++i){
      src[i]=hash_pair<val_type,val_type>{i,(val_type)rand()%10000};
   }

   auto d_hmap=hmap.upload();
   gpu_write<<<1,64>>>(d_hmap, src.data(), src.size());
   cudaDeviceSynchronize();
   hmap.download();

   //Read
   for (const auto& i:hmap){
      std::cout<<"["<<i.first<<" "<<i.second<<"] ";
   }
   std::cout<<std::endl;
}


void advanced_device_usage()
{
   std::cout<<"\nAdvanced Device Usage\n"<<std::endl;
   vector src(64);
   Hashmap<val_type,val_type> hmap;
   hmap.resize(7);
   //Create Input
   for (uint32_t i=0 ; i<64; ++i){
      src[i]=hash_pair<val_type,val_type>{i,(val_type)rand()%10000};
   }

   hmap.insert(src.data(),src.size(),6);
   //Read
   for (const auto& i:hmap){
      std::cout<<"["<<i.first<<" "<<i.second<<"] ";
   }
   std::cout<<std::endl;
}

void basic_hybrid_usage()
{

   std::cout<<"\nHybrid Usage\n"<<std::endl;
   vector src(64);
   Hashmap<val_type,val_type> hmap;
   hmap.resize(7);
   //Create Input
   for (uint32_t i=0 ; i<64; ++i){
      src[i]=hash_pair<val_type,val_type>{i,(val_type)rand()%10000};
   }

   auto d_hmap=hmap.upload();
   gpu_write<<<1,64>>>(d_hmap, src.data(), src.size());
   cudaDeviceSynchronize();
   hmap.download();

   //Read
   for (const auto& i:hmap){
      std::cout<<"["<<i.first<<" "<<i.second<<"] ";
   }
   std::cout<<std::endl;

   d_hmap=hmap.upload();
   gpu_read_and_delete<<<1,64>>>(d_hmap);
   cudaDeviceSynchronize();
   hmap.download();
   std::cout<<"Load factor should be zero and is LF= "<<hmap.load_factor()<<std::endl;
}

int main()
{
   basic_host_usage();
   basic_device_usage();
   advanced_device_usage();
   basic_hybrid_usage();
   return 0;
}

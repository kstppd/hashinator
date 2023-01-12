#include <iostream>
#include <random>
#include "../include/hashinator/hashinator.h"

typedef uint32_t val_type;
using namespace Hashinator;
typedef split::SplitVector<hash_pair<val_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
typedef Hashmap<val_type,val_type> hashmap;



void create_input(vector& src){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<val_type,val_type>& kval=src.at(i);
      kval.first=i;
      kval.second=i;
   }
}


void hashmap_benchmark()
{
   int power=24;
   size_t N = 1<<power;
   vector src(N);
   src.optimizeGPU();
   create_input(src);
   hashmap hmap;
   hmap.insert(src.data(),src.size(),power);
   std::cout<<hmap.load_factor()<<std::endl;
}

int main()
{
   hashmap_benchmark();
   return 0;
}

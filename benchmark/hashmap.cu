#include <iostream>
#include <random>
#include <stdlib.h>
#include "../include/hashinator/hashinator.h"
typedef uint32_t val_type;
using namespace Hashinator;
typedef split::SplitVector<hash_pair<val_type,val_type>,split::split_unified_allocator<hash_pair<val_type,val_type>>,split::split_unified_allocator<size_t>> vector ;
typedef Hashmap<val_type,val_type> hashmap;

void create_input(vector& src){
   for (size_t i=0; i<src.size(); ++i){
      hash_pair<val_type,val_type>& kval=src.at(i);
      kval.first=i;
      kval.second=rand()%100000;
   }
}

void extend_input(vector& src,size_t N){

   auto offset=src.size();
   src.resize(src.size()+N);
   for (uint32_t i=0; i<N; ++i){
      src[i+offset] = hash_pair<val_type,val_type>{(val_type)offset+i,(val_type)offset+i};
   }

}


bool recover_elements(const hashmap& hmap, vector& src){
   for (size_t i=0; i<src.size(); ++i){
      const hash_pair<val_type,val_type>& kval=src.at(i);
      auto retval=hmap.find(kval.first);
      if (retval==hmap.end()){assert(0&& "END FOUND");}
      bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      if (!sane){ 
         return false; 
      }
   }
   return true;
}

void hashmap_benchmark(int power)
{
   size_t N = 1<<power;
   vector src(N);
   create_input(src);
   hashmap hmap;
   for (int i=0; i<10; ++i){
      src.optimizeGPU();
      hmap.insert(src.data(),src.size());
      bool success=recover_elements(hmap,src);
      if (!success){assert(false && "Map is illformed");}
      std::cout<<hmap.load_factor()<<std::endl;
      hmap.clear();
   }
}


int main(int argc, char** argv)
{
   if (argc<2){return 1;}
   int N =  atoi(argv[1]);
   hashmap_benchmark(N);
   return 0;
}


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


void hashmap_benchmark()
{
   int power=24;
   size_t N = 1<<power;
   vector src(N);
   create_input(src);
   src.optimizeGPU();
   hashmap hmap;
   hmap.insert(src.data(),src.size(),power);
   std::cout<<hmap.load_factor()<<std::endl;
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
void hashmap_benchmark_lf()
{
   int power=22;
   int step=1<<18;
   hashmap hmap;
   size_t N = 1<<power;
   vector src(N);
   create_input(src);
   do{
      hmap.clear();
      src.optimizeGPU();
      hmap.insert(src.data(),src.size(),power);
      bool success=recover_elements(hmap,src);
      if (!success){assert(false && "Map is illformed");}
      std::cout<<"Load factor= "<<hmap.load_factor()<<std::endl;
      extend_input(src,step);
   }while(hmap.load_factor()<0.95);
}




int main()
{
   hashmap_benchmark_lf();
   return 0;
}

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "../../src/hashinator_2/hashinator.h"

typedef uint32_t val_type;

void load_N_elems(Hashinator<val_type,val_type>& map ,size_t N){
   for (size_t i =0; i<N;i++){
      map[i] = i;
   }
}

void load_N_elems_2(Hashinator<val_type,val_type>& map ,size_t N){
   for (size_t i =0; i<N;i++){
      map.at(i) = 2*i;
   }
}

void test1(){
   Hashinator<val_type,val_type> map;
   map.print_bank();   
   load_N_elems(map,128);
   map.print_all();
   map.print_bank();   
   map.clear();
   map.print_all();
   map.print_bank();   
   load_N_elems(map,256);
   map.print_all();
   map.print_bank();   


   for (auto kval=map.begin(); kval!=map.end();++kval){
      std::cout<<  kval->first<<" "<<kval->second <<std::endl;
      map.erase(kval);
   }
   map.print_all();
   map.print_bank();   
   load_N_elems_2(map,12);

   for (auto kval=map.begin(); kval!=map.end();kval++){
      std::cout<<  kval->first<<" "<<kval->second <<std::endl;
   }

   auto elem=map.find(10);
   if (elem==map.end()){std::cout<<"Element does not exist"<<std::endl;}
   if (elem!=map.end()){std::cout<<"Element exists at "<<elem.getIndex() <<std::endl;}



}

int main(){
   test1();
}


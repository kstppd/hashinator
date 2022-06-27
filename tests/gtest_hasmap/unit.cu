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

void test1(){
   Hashinator<val_type,val_type> map;
   load_N_elems(map,1024);
   map.print_all();
}

int main(){
   test1();
}


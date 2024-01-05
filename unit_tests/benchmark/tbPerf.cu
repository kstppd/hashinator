#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <unordered_set>
#include <random>
#include "../../include/hashinator/hashinator.h"
static constexpr int R = 10;

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef uint32_t key_type;
typedef split::SplitVector<hash_pair<key_type,val_type>> vector ;
typedef split::SplitVector<key_type> key_vec;
typedef split::SplitVector<val_type> val_vec;
using hashmap= Hashmap<key_type,val_type>;

auto generateNonDuplicatePairs(vector& src,const size_t size)->void {
    std::unordered_set<int> keys;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<val_type> dist(1, std::numeric_limits<val_type>::max());

    src.clear();
    while (src.size() < size) {
        val_type key = dist(gen);
        // Check if the key is already present
        if (keys.find(key) == keys.end()) {
           val_type val=static_cast<val_type>(key/2);
            src.push_back({key,val});
            keys.insert(key);
        }
    }
}


template <class Fn, class ... Args>
auto timeMe(Fn fn, Args && ... args){
   std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   double total_time=0;
   start = std::chrono::high_resolution_clock::now();
   fn(args...);
   stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop- start).count();
   total_time+=duration;
   return total_time;
}

enum class METHOD{
   NOOP,
   REHASH,
   CLEAN
};

void precondition(hashmap& hmap,vector& src,key_vec& keys, val_vec& vals,size_t size){
   hmap.insert(src.data(),0.9*size,1);
   hmap.erase(keys.data(),0.2*(size));
   (void)keys;
   (void)vals;
}

void insert_control(hashmap& hmap,vector& src,key_vec& keys, val_vec& vals,size_t size,METHOD method){
   switch (method){
      case METHOD::NOOP:
         break;
      case METHOD::REHASH:
         hmap.device_rehash(std::log2(size));
         break;
      case METHOD::CLEAN:
         hmap.clean_tombstones();
         break;
      default:
         assert(0 && "No method selected!");
   }
   (void)keys;
   (void)vals;
   hmap.insert(src.data(),0.1*size,1);
}

double test(int sz,vector& cpu_src,key_vec& keyBuffer,val_vec& valBuffer,METHOD method){
   hashmap hmap(sz);
   hmap.optimizeGPU();
   cpu_src.optimizeGPU();
   keyBuffer.optimizeGPU();
   valBuffer.optimizeGPU();
   double t={0};
   for (int i =0; i<R; i++){
      hmap.optimizeGPU();
      keyBuffer.optimizeGPU();
      valBuffer.optimizeGPU();
      cpu_src.optimizeGPU();
      precondition(hmap,cpu_src,keyBuffer,valBuffer,cpu_src.size());
      t+=timeMe(insert_control,hmap,cpu_src,keyBuffer,valBuffer,cpu_src.size(),method);
      hmap.clear();
   }
   return t/R;
}


int main(){
   printf("Results for Control Sizepower-- Device Rehash -- Tombstone Cleaning\n");
   for (int sz=10; sz<=20;sz++){
      vector cpu_src;
      key_vec keyBuffer;
      val_vec valBuffer;
      generateNonDuplicatePairs(cpu_src,(1<<sz));
      for (auto& i : cpu_src){
         keyBuffer.push_back(i.first);
         valBuffer.push_back(i.first);
      }
      auto time_control = test(sz,cpu_src,keyBuffer,valBuffer,METHOD::NOOP);
      auto time_rehash = test(sz,cpu_src,keyBuffer,valBuffer,METHOD::REHASH);
      auto time_clean = test(sz,cpu_src,keyBuffer,valBuffer,METHOD::CLEAN);
      printf("%d \t %.03f %.f %.03f \n",sz,time_control,time_rehash,time_clean);
   }
   return 0;
}

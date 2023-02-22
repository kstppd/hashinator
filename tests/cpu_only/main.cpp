#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../../include/hashinator/hashinator.h"
#include <gtest/gtest.h>




#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ

using namespace std::chrono;
using namespace Hashinator;
typedef uint32_t val_type;
typedef split::SplitVector<cuda::std::pair<val_type,val_type>> vector ;
// typedef Hashmap<val_type,val_type> hashmap;


//template <class Fn, class ... Args>
//auto execute_and_time(const char* name,Fn fn, Args && ... args) ->bool{
   //std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> start,stop;
   //double total_time=0;
   //start = std::chrono::high_resolution_clock::now();
   //bool retval=fn(args...);
   //stop = std::chrono::high_resolution_clock::now();
   //auto duration = duration_cast<microseconds>(stop- start).count();
   //total_time+=duration;
   //std::cout<<name<<" took "<<total_time<<" us"<<std::endl;
   //return retval;
//}


//void create_input(vector& src, uint32_t bias=0){
   //for (size_t i=0; i<src.size(); ++i){
      //cuda::std::pair<val_type,val_type>& kval=src.at(i);
      //kval.first=i + bias;
      //kval.second=rand()%1000000;
   //}
//}

//void create_random_input(vector& src){
   //for (size_t i=0; i<src.size(); ++i){
      //cuda::std::pair<val_type,val_type>& kval=src.at(i);
      //kval.first=(rand());
      //kval.second=rand()%1000000;
   //}
//}


//void cpu_write(hashmap& hmap, vector& src){
   //for (size_t i=0; i<src.size(); ++i){
      //const cuda::std::pair<val_type,val_type>& kval=src.at(i);
      //hmap.at(kval.first)=kval.second;
   //}
//}

//bool recover_elements(const hashmap& hmap, vector& src){
   //for (size_t i=0; i<src.size(); ++i){
      //const cuda::std::pair<val_type,val_type>& kval=src.at(i);
      //auto retval=hmap.find(kval.first);
      //if (retval==hmap.end()){assert(0&& "END FOUND");}
      //bool sane=retval->first==kval.first  &&  retval->second== kval.second ;
      //if (!sane){ 
         //return false; 
      //}
   //}
   //return true;
//}

//bool test_hashmap_1(val_type power){
   //size_t N = 1<<power;



//}

//TEST(HashmapUnitTets , Host_Device_Insert_Delete_Global_Tets){
   //for (int power=5; power<22; ++power){
      //std::string name= "Power= "+std::to_string(power);
      //bool retval = execute_and_time(name.c_str(),test_hashmap_1 ,power);
      //expect_true(retval);
   //}
//}

int main(int argc, char* argv[]){
   srand(time(NULL));
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

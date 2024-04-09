#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include "../../../include/splitvector/splitvec.h"
#include "../../../include/splitvector/split_tools.h"
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define TARGET 1

typedef uint32_t int_type ;
typedef struct{
   int_type num;
   int_type flag;
} test_t;
typedef split::SplitVector<test_t> vector; 
size_t count = 0;

void print_vector(vector& v){
   std::cout<<"-------------------"<<std::endl;
   std::cout<<"Size = "<<v.size()<<std::endl;;
   for (const auto& i:v){
      std::cout<<"["<<i.num<<","<<i.flag<<"] ";
   }
   std::cout<<"\n-------------------"<<std::endl;
   std::cout<<std::endl;
}

void fill_vec(vector& v, size_t targetSize){
   count=0;
   size_t st=0;
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<int_type> dist(1, std::numeric_limits<int_type>::max());
   v.clear();
   while (v.size() < targetSize) {
      int_type val =++st;// dist(gen);
      v.push_back(test_t{val,(val%2==0)});
      if (val%2 == 0){count++;};
    }
}

void fill_vec_lin(vector& v, size_t targetSize){
   v.clear();
   int_type s=0;
   while (v.size() < targetSize) {
      v.push_back(test_t{s,s});
      s++;
    }
}

bool checkFlags(const vector& v,const int_type target){
   for (const auto& i:v){
      if (i.flag!=target){return false;}
   }
   return true;
}

bool normal_compactions(int power){
   vector v;
   fill_vec(v,1<<power);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
   split::tools::copy_if(v,output1,predicate_on);
   split::tools::copy_if(v,output2,predicate_off);
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   return sane1 && sane2 && sane3 && sane4 && sane5;
}

//In this example we use the preallocated compaction one after the other in a serial fashion
bool preallocated_compactions_basic(int power){
   vector v;
   fill_vec(v,1<<power);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
   
   //Here we determine how much memory we need for one compaction on v.
   size_t bytesNeeded=split::tools::estimateMemoryForCompaction(v);
   
   /*
     Since we now know the memory needed  let's allocate a buffer with that size. 
     This can be done with mallocAsync, malloc, or managedMalloc but anyways let's 
     do old good device memory for now
    */

   void* buffer=nullptr;
   SPLIT_CHECK_ERR (split_gpuMalloc( (void**)&buffer , bytesNeeded));

   //These mempools are now allocation free. They essentially just manage the buffer correclty!
   //Please !!ALWAYS!! forwarding here to preserve move semantics as the pool might change later on
   split::tools::copy_if(v,output1,predicate_on,
                           std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{buffer,bytesNeeded}));

   split::tools::copy_if(v,output2,predicate_off,
                           std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{buffer,bytesNeeded}));
   //Deallocate our good buffer
   SPLIT_CHECK_ERR (split_gpuFree(buffer));
   
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   return sane1 && sane2 && sane3 && sane4 && sane5;
}

//In this example we use the preallocated compaction one after the other in a serial fashion
bool preallocated_compactions_basic_overload(int power){
   vector v;
   fill_vec(v,1<<power);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
   
   //Here we determine how much memory we need for one compaction on v.
   size_t bytesNeeded=split::tools::estimateMemoryForCompaction(v);
   
   /*
     Since we now know the memory needed  let's allocate a buffer with that size. 
     This can be done with mallocAsync, malloc, or managedMalloc but anyways let's 
     do old good device memory for now
    */

   void* buffer=nullptr;
   SPLIT_CHECK_ERR (split_gpuMalloc( (void**)&buffer , bytesNeeded));

   //These mempools are now allocation free. They essentially just manage the buffer correclty!
   //Please !!ALWAYS!! forwarding here to preserve move semantics as the pool might change later on
   split::tools::copy_if(v,output1,predicate_on,buffer,bytesNeeded);
   split::tools::copy_if(v,output2,predicate_off,buffer,bytesNeeded);
   //Deallocate our good buffer
   SPLIT_CHECK_ERR (split_gpuFree(buffer));
   
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   return sane1 && sane2 && sane3 && sane4 && sane5;
}

//In this example we use the preallocated compaction to perform two compaction in parallel using different streams
bool preallocated_compactions_medium(int power){
   vector v;
   fill_vec(v,1<<power);
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   vector output1(v.size());
   vector output2(v.size());
   
   //Here we determine how much memory we need for one compaction on v.
   size_t bytesNeeded=split::tools::estimateMemoryForCompaction(v);
   
   /*
     Since we now know the memory needed  let's allocate a buffer with that size. 
     This can be done with mallocAsync, malloc, or managedMalloc but anyways let's 
     do old good device memory for now
    */


   /*
      NOTE: I ALLOCATE DOUBLE THE SIZE HERE BECAUSE WE NEED THE 2 COMPACTIONS HAPPENING IN PARALLEL
   */
   void* buffer=nullptr;
   SPLIT_CHECK_ERR (split_gpuMalloc( (void**)&buffer ,2*bytesNeeded));
   
   //Let's create two streams damnit!
   std::array<split_gpuStream_t,2> streams;
   for (auto& s:streams){
      SPLIT_CHECK_ERR( split_gpuStreamCreate( &s ));
   }


   //These mempools are now allocation free. They essentially just manage the buffer correclty!
   //Please !!ALWAYS!! forwarding here to preserve move semantics as the pool might change later on
   
   //This guy goes on stream 1
   split::tools::copy_if(v,output1,predicate_on,
                           std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{buffer,bytesNeeded}),streams[0]);

   //This guy goes on stream 2
   /*
      NOTE: THE BUFFER HERE IS OFFSETED!!!
   */
   void* start = reinterpret_cast<void*> ( reinterpret_cast<char*>(buffer)+bytesNeeded);
   split::tools::copy_if(v,output2,predicate_off,
                           std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{start,bytesNeeded}),streams[1]);


   //Wait for them!
   for (auto s:streams){
      SPLIT_CHECK_ERR( split_gpuStreamSynchronize( s ));
   }

   //Destroy streams
   for (auto s:streams){
      SPLIT_CHECK_ERR( split_gpuStreamDestroy( s ));
   }
   //Deallocate our good buffer
   SPLIT_CHECK_ERR (split_gpuFree(buffer));
   
   bool sane1 = checkFlags(output1,1);
   bool sane2 = checkFlags(output2,0);
   bool sane3 = ((output1.size()+output2.size())==v.size());
   bool sane4 =(  output1.size() ==count );
   bool sane5 = ( output2.size() ==v.size()-count );
   return sane1 && sane2 && sane3 && sane4 && sane5;
}


//In this example we  go HAM and use a parallel region to compacty 1K splitvectors because we can!
bool preallocated_compactions_HAM(int power){

   constexpr size_t N =  512;
   constexpr size_t nThreads =  8;
   constexpr size_t streamsPerThread =  2;
   
   std::array<vector,N> vecs;
   std::array<vector,N> out1;
   std::array<vector,N> out2;

   //Fill them up
   for (auto& v:vecs){
      fill_vec(v,1<<power);
   }
   //Don't mind these 2, they are for the unit testing
   for (auto& v:out1){
      v.resize(1<<power);
   }
   for (auto& v:out2){
      v.resize(1<<power);
   }

   //Prepare our predicates
   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   
   //Here we determine how much memory we need for one compactions.
   size_t bytesNeeded=split::tools::estimateMemoryForCompaction(vecs[0]);
   
   //Here we allocate a buffer that can fit all the compactions at once so streamsPerThread*nThreads
   void* buffer=nullptr;
   SPLIT_CHECK_ERR (split_gpuMalloc( (void**)&buffer ,streamsPerThread*nThreads*bytesNeeded));

   
   //Just restrict to 8 threads and no dynamic teams to make this test the same every time!
   omp_set_dynamic(0);     
   omp_set_num_threads(nThreads);

   for (size_t i =0 ; i<vecs.size();i++){
       vecs[i].optimizeGPU();
       out1[i].optimizeGPU();
       out2[i].optimizeGPU();
   }
   SPLIT_CHECK_ERR( split_gpuDeviceSynchronize( ));

   //Compact away!!
   #pragma omp parallel for 
   for (size_t i =0 ; i<vecs.size();i++){
      const auto tid=omp_get_thread_num();
      void* tidIndex_1 = reinterpret_cast<void*> ( reinterpret_cast<char*>(buffer)+tid*streamsPerThread*bytesNeeded);
      void* tidIndex_2 = reinterpret_cast<void*> ( reinterpret_cast<char*>(buffer)+tid*streamsPerThread*bytesNeeded+bytesNeeded);
      split::tools::copy_if(vecs[i],out1[i],predicate_on,
                              std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{tidIndex_1,bytesNeeded}));

      split::tools::copy_if(vecs[i],out2[i],predicate_off,
                              std::forward<split::tools::splitStackArena>(split::tools::splitStackArena{tidIndex_2,bytesNeeded}));
   }


   SPLIT_CHECK_ERR( split_gpuDeviceSynchronize( ));

   //Deallocate our good buffer
   SPLIT_CHECK_ERR (split_gpuFree(buffer));
   
   //Now let's verify 
   bool sane1 =true;
   bool sane2 =true;
   bool sane3 =true;
   for (size_t i = 0; i < vecs.size();i++){
      sane1&= checkFlags(out1[i],1);
      sane2&= checkFlags(out2[i],0);
      sane3&= ((out1[i].size()+out2[i].size())==vecs[i].size());
      if (sane1&&sane2&&sane3==false){
         std::cout<<out1[i].size()<<" "<<out2[i].size()<<" "<<vecs[i].size()<<std::endl;
         assert(false);
      }
   }
   return sane1 && sane2 && sane3 ;
}

TEST(StremCompaction , Compaction_Simple){
   for (uint32_t i =5; i< 15; i++){
      expect_true(normal_compactions(i));
   }
}

TEST(StremCompaction , Compaction_Preallocated_Basic){
   for (uint32_t i =5; i< 15; i++){
      expect_true(preallocated_compactions_basic(i));
   }
}

TEST(StremCompaction , Compaction_Preallocated_Basic_Overload){
   for (uint32_t i =5; i< 15; i++){
      expect_true(preallocated_compactions_basic_overload(i));
   }
}

TEST(StremCompaction , Compaction_Preallocated_Medium){
   for (uint32_t i =5; i< 15; i++){
      expect_true(preallocated_compactions_medium(i));
   }
}

TEST(StremCompaction , Compaction_Preallocated_HAM){
   for (uint32_t i =5; i< 15; i++){
      expect_true(preallocated_compactions_HAM(i));
   }
}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

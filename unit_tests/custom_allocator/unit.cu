#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include "../../include/splitvector/splitvec.h"
#include "../../include/splitvector/split_tools.h"
#include "../../include/common.h"
#include "../../include/splitvector/archMacros.h"
#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
#define TARGET 1

/**
 * @brief Custom allocator for unified memory (GPU and CPU accessible).
 *
 * This class provides an allocator for unified memory, which can be accessed
 * by both the GPU and the CPU. It allocates and deallocates memory using split_gpuMallocManaged
 * and split_gpuFree functions, while also providing constructors and destructors for objects.
 *
 * @tparam T Type of the allocated objects.
 */
template <class T>
class customAllocator {
public:
   typedef T value_type;
   typedef value_type* pointer;
   typedef const value_type* const_pointer;
   typedef value_type& reference;
   typedef const value_type& const_reference;
   typedef ptrdiff_t difference_type;
   typedef size_t size_type;
   template <class U>
   struct rebind {
      typedef customAllocator<U> other;
   };
   /**
    * @brief Default constructor.
    */
   customAllocator() throw() {}

   /**
    * @brief Copy constructor with different type.
    */
   template <class U>
   customAllocator(customAllocator<U> const&) throw() {}
   pointer address(reference x) const { return &x; }
   const_pointer address(const_reference x) const { return &x; }

   pointer allocate(size_type n, const void* /*hint*/ = 0) {
      T* ret;
      assert(n && "allocate 0");
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n * sizeof(value_type)));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   static void* allocate_raw(size_type n, const void* /*hint*/ = 0) {
      void* ret;
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   void deallocate(pointer p, size_type n) {
      if (n != 0 && p != 0) {
         SPLIT_CHECK_ERR(split_gpuFree(p));
      }
   }
   static void deallocate(void* p, size_type n) {
      if (n != 0 && p != 0) {
         SPLIT_CHECK_ERR(split_gpuFree(p));
      }
   }

   size_type max_size() const throw() {
      size_type max = static_cast<size_type>(-1) / sizeof(value_type);
      return (max > 0 ? max : 1);
   }

   template <typename U, typename... Args>
   __host__ __device__ void construct(U* p, Args&&... args) {
      ::new (p) U(std::forward<Args>(args)...);
   }

   void destroy(pointer p) { p->~value_type(); }
};

typedef uint32_t int_type ;
typedef struct{
   int_type num;
   int_type flag;
} test_t;
typedef split::SplitVector<test_t> vector;
typedef split::SplitVector<test_t,customAllocator<test_t>> customAllocatorVector; 
size_t count = 0;

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

bool checkFlags(const customAllocatorVector& v,const int_type target){
   for (const auto& i:v){
      if (i.flag!=target){return false;}
   }
   return true;
}

bool run_test_small_loop_variant(size_t size){
   // std::cout<<"Testing with vector size: "<<size<<std::endl;
   vector* v=new vector();
   fill_vec(*v,size);

   auto predicate_on =[]__host__ __device__ (test_t element)->bool{ return element.flag == 1 ;};
   auto predicate_off =[]__host__ __device__ (test_t element)->bool{ return element.flag == 0 ;};
   customAllocatorVector* output1=new customAllocatorVector(nextPow2(2*v->size()));
   customAllocatorVector* output2=new customAllocatorVector(nextPow2(2*v->size()));

   split::tools::copy_if_loop(*v,*output1,predicate_on);
   split::tools::copy_if_loop(*v,*output2,predicate_off);
   SPLIT_CHECK_ERR( split_gpuDeviceSynchronize() );

   bool sane1 = checkFlags(*output1,1);
   bool sane2 = checkFlags(*output2,0);
   bool sane3 = ((output1->size()+output2->size())==v->size());
   bool sane4 =(  output1->size() ==count );
   bool sane5 = ( output2->size() ==v->size()-count );
   // printf( " %d - %d - %d - %d - %d\n",sane1,sane2,sane3,sane4,sane5 );  
   bool retval =  sane1 && sane2 && sane3 && sane4 && sane5;
   return retval;
}

TEST(StremCompaction , Compaction_Tests_Linear_Loop_Variant){
   for (size_t s=32; s< 1024; s++ ){
      bool a = run_test_small_loop_variant(s);
      expect_true(a);
   }

}

int main(int argc, char* argv[]){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

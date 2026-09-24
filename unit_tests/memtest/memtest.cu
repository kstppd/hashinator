#include "memtest.h"

size_t base_usage = 0;
size_t mem_limit = 6 * std::pow(1024,3);

/* Memory reporting function
 */
void gpu_reportMemory(const std::vector<testStructure> &objs) {
   size_t o_size=0;
   size_t o_cap=0;
   const size_t nob=objs.size();
   for (uint i=0; i<nob; ++i) {
      o_size += (objs.at(i)).sizeInBytes();
      o_cap += (objs.at(i)).capacityInBytes();
   }

   size_t free_byte ;
   size_t total_byte ;
   SPLIT_CHECK_ERR( cudaMemGetInfo( &free_byte, &total_byte) );
   size_t used_mb = (total_byte-free_byte)/(1024*1024);

   // store base usage at start
   if (nob==0) {
//      base_usage = used_mb;
      std::cerr<<" =================================="<<std::endl;
      std::cerr<<" GPU Memory: base usage is "<<base_usage<<" Mbytes"<<std::endl;
      std::cerr<<" =================================="<<std::endl;
      return;
   }
   int64_t int_used_mb = (int64_t)used_mb - (int64_t)base_usage;
   std::cerr<<" =================================="<<std::endl;
   std::cerr<<" GPU Memory report"<<std::endl;
   std::cerr<<"   objects size:            "<<o_size/(1024*1024)<<" Mbytes"<<std::endl;
   std::cerr<<"   objects capacity:        "<<o_cap/(1024*1024)<<" Mbytes"<<std::endl;
   std::cerr<<"   Reported Hardware use:   "<<int_used_mb<<" Mbytes"<<std::endl;
   std::cerr<<" =================================="<<std::endl;
   return;
}

TEST(Test_GPU,Memory) {
   //int myDevice;
   const int n_objs = 20;
   const int n_loops = 10;

   std::vector<testStructure> storage;

   gpu_reportMemory(storage);

   //SPLIT_CHECK_ERR( gpuGetDevice(&myDevice) );
   for (uint i=0; i<n_objs; ++i) {
      storage.push_back(testStructure(1024*1024));
   }

   const int initial_size = storage[0].size();
   gpu_reportMemory(storage);

   //final_size = std::pow(2,35) // 34 gigs
   for (uint j=0; j<n_loops; ++j) {
      size_t newSize = initial_size * std::pow(2,j);

      if (n_objs * newSize > mem_limit) {
         break;
      }
      std::cerr<<std::endl<<std::endl;
      std::cerr<<std::endl<<std::endl;
      std::cerr<<"============  CYCLE "<<j<<" ==============="<<std::endl;

/*
      std::cerr<<" recapacitate "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].recapacitate(newSize);
      }
      gpu_reportMemory(storage);

      std::cerr<<" resize "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].resize(newSize-1);
      }
      gpu_reportMemory(storage);

      std::cerr<<" resize down "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].resize(newSize/1024);
      }
      gpu_reportMemory(storage);

      std::cerr<<" shrink_to_fit 1 "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].shrink_to_fit();
      }
      gpu_reportMemory(storage);

      std::cerr<<std::endl<<std::endl;
*/
      std::cerr<<" recapacitate again "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].recapacitate(newSize);
      }
      gpu_reportMemory(storage);

      std::cerr<<" resize again "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].resize(newSize-1);
      }
      gpu_reportMemory(storage);
      std::cerr<<" resize down again "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].resize(newSize/1024);
      }
      gpu_reportMemory(storage);
      std::cerr<<" shrink_to_fit 2 "<<j<<std::endl;
      for (uint i=0; i<n_objs; ++i) {
         storage[i].shrink_to_fit_2();
         //storage[i].shrink_to_fit();
      }
      gpu_reportMemory(storage);
}

   SPLIT_CHECK_ERR( split_gpuDeviceSynchronize() );
   EXPECT_TRUE(true);
}
__host__ int main(int argc, char* argv[]) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

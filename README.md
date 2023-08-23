## Hashinator: A hybrid hashmap designed for heterogeneous computing.

+ Hashinator is a header only hashmap implementation designed to work on both CPU and GPU architectures. It does so by utilizing the [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) model. At this point hashinator NVIDIA and AMD GPUs.  

+ Hashinator's developement was motivated by Vlasiator's porting to GPUs. To learn more about that visit [Vlasiator's github page](https://github.com/fmihpc/vlasiator).

+ Hashinator uses a custom vector implementation named SplitVector which is hosted in this repository as well. Splitvector takes the burden of memory management away from hashinator and provides a linear buffer of Unified Memory with proper prefetching routines. Its API is made to resemble that of **std::vector** for easy integration with already existing codes. It can be used as a standalone container and does not depend on Hashinator.

+ Hashinator uses an open addressing scheme together withe the Fibonnacci multiplicatve hash function to hash key-value into a contigious buffer. Key-value pairs can be inserted, querried and deleted via three different APIs. The *host-only* API performs all operation on the CPU in a serial manner. The *device-only* API  performs operations from device code and the *accelerated* API utilizes the GPU to performs operation in parallel.

+ The *accelerated* API uses a parallel probing scheme inspired by [Warpcore](https://github.com/sleeepyjack/warpcore), however using a custom implementation that does not leverage [Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/).

+ A novel tombtone cleaning method is provided with Hashinator that allowes tombstones to be removed from the hashmap in parallel using the GPU.

+ Hashinator and SplitVector are arch agnostic. The codebase can be compiled with NVCC or ROCm without the need of hipification.  

+ For systems without GPUs, Hashinator and SplitVector compile with a c++ compiler by defining ```-DHASHINATOR_HOST_ONLY``` and ```-DSPLIT_HOST_ONLY``` respectively.

## Installation
No installation required. Just include  "hashinator.h" . However, if you plan to use Hashinator at its full potential you will need a system with a dedicated GPU card, either NVIDIA or AMD and  a healthy installation of CUDA or ROCm. Hashinator rerquires at least ```cuda-9.0``` or ```rocm 5.4```. To run the tests ```googletest``` needs to be installed.

## Example Usage: 
### SplitVector: Basic Usage on host  
```c++
#include "splitvec.h"
//main.cpp
int main()
{
   //main.cpp
   using vector = split::SplitVector<int>;

   split::SplitVector<int>  vec {1,2,3,4,5};
   vec.push_back(4);
   std::cout<<vec[3]<<std::endl;
   vec.clear();
}
```
`g++   main.cpp  -std=c++17 -o example`

### SplitVector: Basic Usage on device
```
#include "splitvec.h"
//main.cu
using vector = split::SplitVector<int>;

void push_back_kernel(vector* a){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	a->device_push_back(index);
}

int main()
{
	vector* vec = new vector{1,2,3,4,5};
	vec->reserve(128);
	vec->optimizeGPU();
	push_back_kernel<<<1,64>>>(vec);
	cudaDeviceSynchronize();
	vec->optimizeCPU();
	std::cout<<*vec<<std::endl;
	delete vec;
}
```
`nvcc   main.cu  -std=c++17  --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -o example`

### Hashinator: Basic Usage on host

```c++
//main.cpp
#include "hashinator.h"

int main()
{

   Hashmap<uint_32t,uint32_t> hmap;

   //Write
   for (uint32_t i=0 ; i<64; ++i){
      hmap[i]=rand()%10000;
   }

   //Read
   for (const auto& i:hmap){
      std::cout<<"["<<i.first<<" "<<i.second<<"] ";
   }
   std::cout<<std::endl;
}
```
`g++   main.cpp  -std=c++17 -o example`
### Hashinator: Basic usage on device

```c++
//main.cu
#include "hashinator.h"

__global__
void gpu_write(Hashmap<val_type,val_type>* hmap, hash_pair<val_type,val_type>*src, size_t N)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N ){
	  hmap->set_element(src[index].first, src[index].second);
	}
}

int main()
{
	std::cout<<"\nDevice Usage\n"<<std::endl;
	Hashmap<val_type,val_type>* hmap=new Hashmap<val_type,val_type>;
	hmap->resize(7);
	//Create Input
	for (uint32_t i=0 ; i<64; ++i){
	  src[i]=hash_pair<val_type,val_type>{i,(val_type)rand()%10000};
	}

	gpu_write<<<1,64>>>(hmap, src.data(), src.size());
	cudaDeviceSynchronize();
}
```
`nvcc   main.cu  -std=c++17  --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -o example`

### Hashinator: Basic usage of the accelerated mode
```c++
//main.cu
#include "hashinator.h"

int main()
{
	vector src(64);
	Hashmap<val_type,val_type> hmap;
	hmap.resize(7);
	
	//Create Input
	for (uint32_t i=0 ; i<64; ++i){
	  src[i]=hash_pair<val_type,val_type>{i,(val_type)rand()%10000};
	}

	//Insert using the accelerated mode
	hmap.insert(src.data(),src.size());
}
```
`nvcc   main.cu  -std=c++17  --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -o example`

You can have a look in the Doxygen for a more feature-rich explanation of the methods and tools included!   

## Test Coverage
Hashinator and SplitVector include a suite of unit tests using [googletest](https://github.com/google/googletest) which live under the ```tests``` directory. These tests try to cover as many features as possible to avoid the silent introduction of bugs! The units tests automatically trigger for pushes to the dev and master branches. 

#ifdef __NVCC__
#define  HOST_DEVICE  __host__ __device__
#define  HOST         __host__
#define  DEVICE       __device__ 
#else
#define  HOST_DEVICE  __host__ __device__  
#define  HOST         __host__   
#define  DEVICE       __device__  
#endif



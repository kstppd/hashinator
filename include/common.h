#pragma once
#ifdef __NVCC__
#define  HOST_DEVICE  __host__ __device__
#define  HOST         __host__
#define  DEVICE       __device__ 
#else
#define  HOST_DEVICE  
#define  HOST         
#define  DEVICE       
#endif

#define HW_MAXBLOCKSIZE 1024
#define HW_WARPSIZE 32
#define HW_SM_BANKS 32
#define HW_SM_BANKS_LOG 5
#define HW_VOTING_MASK 0xFFFFFFFF

/* File:    gpuMacros.h
 * Authors: Kostis Papadakis (2023)
 *
 * Defines common macros for CUDA and HIP
 *
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 * */

#pragma once
/* Select the compiled architecture */
#ifdef __CUDACC__

/* Needed here (not just transitively via gpu_wrappers.h, which some
 * translation units include after this file) so CUDART_VERSION and the
 * cudaMemLocation-related types below are guaranteed to be visible. */
#include <cuda_runtime_api.h>

#define split_gpuGetLastError cudaGetLastError
#define split_gpuGetErrorString cudaGetErrorString
#define split_gpuPeekAtLastError cudaPeekAtLastError

#define split_gpuSetDevice cudaSetDevice
#define split_gpuGetDevice cudaGetDevice
#define split_gpuGetDeviceCount cudaGetDeviceCount
#define split_gpuGetDeviceProperties cudaGetDeviceProperties
#define split_gpuDeviceSynchronize cudaDeviceSynchronize
#define split_gpuDeviceReset cudaDeviceReset

#define split_gpuFree cudaFree
#define split_gpuFreeHost cudaFreeHost
#define split_gpuFreeAsync cudaFreeAsync
#define split_gpuMalloc cudaMalloc
#define split_gpuMallocHost cudaMallocHost
#define split_gpuMallocAsync cudaMallocAsync
#define split_gpuMallocManaged cudaMallocManaged
// this goes to cudaMallocHost because we don't support flags
#define split_gpuHostAlloc cudaMallocHost
#define split_gpuHostAllocPortable cudaHostAllocPortable
#define split_gpuMemcpy cudaMemcpy
#define split_gpuMemcpyAsync cudaMemcpyAsync
#define split_gpuMemset cudaMemset
#define split_gpuMemsetAsync cudaMemsetAsync

#define split_gpuMemAdviseSetAccessedBy cudaMemAdviseSetAccessedBy
#define split_gpuMemAdviseSetPreferredLocation cudaMemAdviseSetPreferredLocation
#define split_gpuMemAttachSingle cudaMemAttachSingle
#define split_gpuMemAttachGlobal cudaMemAttachGlobal

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
/* CUDA 13 dropped the (ptr, count, int device, stream) overloads of
 * cudaMemPrefetchAsync/cudaMemAdvise in favor of a cudaMemLocation-based
 * signature. Keep call sites on the old int-device form and translate here. */
static inline cudaMemLocation split_gpuMakeMemLocation(int device) {
   cudaMemLocation location;
   location.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
   location.id = (device == cudaCpuDeviceId) ? 0 : device;
   return location;
}
static inline cudaError_t split_gpuMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
   return cudaMemPrefetchAsync(devPtr, count, split_gpuMakeMemLocation(dstDevice), 0, stream);
}
static inline cudaError_t split_gpuMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
   return cudaMemAdvise(devPtr, count, advice, split_gpuMakeMemLocation(device));
}
#define split_gpuMemAdviseDeclared
#else
#define split_gpuMemPrefetchAsync cudaMemPrefetchAsync
#endif

#define split_gpuStreamCreate cudaStreamCreate
#define split_gpuStreamDestroy cudaStreamDestroy
#define split_gpuStreamWaitEvent cudaStreamWaitEvent
#define split_gpuStreamSynchronize cudaStreamSynchronize
#define split_gpuStreamAttachMemAsync cudaStreamAttachMemAsync
#define split_gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define split_gpuStreamCreateWithPriority cudaStreamCreateWithPriority
#define split_gpuStreamDefault cudaStreamDefault

#define split_gpuEventCreate cudaEventCreate
#define split_gpuEventCreateWithFlags cudaEventCreateWithFlags
#define split_gpuEventDestroy cudaEventDestroy
#define split_gpuEventQuery cudaEventQuery
#define split_gpuEventRecord cudaEventRecord
#define split_gpuEventSynchronize cudaEventSynchronize
#define split_gpuEventElapsedTime cudaEventElapsedTime

/* driver_types */
#define split_gpuError_t cudaError_t
#define split_gpuSuccess cudaSuccess

#define split_gpuStream_t cudaStream_t
#define split_gpuDeviceProp cudaDeviceProp_t

#define split_gpuEvent_t cudaEvent_t
#define split_gpuEventDefault cudaEventDefault
#define split_gpuEventBlockingSync cudaEventBlockingSync
#define split_gpuEventDisableTiming cudaEventDisableTiming

#define split_gpuMemcpyKind cudaMemcpyKind
#define split_gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define split_gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define split_gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define split_gpuMemcpyToSymbol cudaMemcpyToSymbol

#define split_gpuCpuDeviceId cudaCpuDeviceId
#define split_gpuMemoryAdvise cudaMemoryAdvise
#ifndef split_gpuMemAdviseDeclared
#define split_gpuMemAdvise cudaMemAdvise
#endif

#elif __HIP__

#define split_gpuGetLastError hipGetLastError
#define split_gpuGetErrorString hipGetErrorString
#define split_gpuPeekAtLastError hipPeekAtLastError
#define split_gpuSetDevice hipSetDevice
#define split_gpuGetDevice hipGetDevice
#define split_gpuGetDeviceCount hipGetDeviceCount
#define split_gpuGetDeviceProperties hipGetDeviceProperties
#define split_gpuDeviceSynchronize hipDeviceSynchronize
#define split_gpuDeviceReset hipDeviceReset
#define split_gpuFree hipFree
#define split_gpuFreeHost hipHostFree
#define split_gpuFreeAsync(ptr,stream) hipFree(ptr)
#define split_gpuMalloc hipMalloc
#define split_gpuMallocHost hipHostMalloc
#define split_gpuMallocAsync hipMallocAsync
#define split_gpuMallocManaged hipMallocManaged
#define split_gpuHostAlloc hipHostMalloc
#define split_gpuHostAllocPortable hipHostAllocPortable
#define split_gpuMemcpy hipMemcpy
#define split_gpuMemcpyAsync hipMemcpyAsync
#define split_gpuMemset hipMemset
#define split_gpuMemsetAsync hipMemsetAsync
#define split_gpuMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define split_gpuMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define split_gpuMemAttachSingle hipMemAttachSingle
#define split_gpuMemAttachGlobal hipMemAttachGlobal
#define split_gpuMemPrefetchAsync hipMemPrefetchAsync
#define split_gpuStreamCreate hipStreamCreate
#define split_gpuStreamDestroy hipStreamDestroy
#define split_gpuStreamWaitEvent hipStreamWaitEvent
#define split_gpuStreamSynchronize hipStreamSynchronize
#define split_gpuStreamAttachMemAsync hipStreamAttachMemAsync
#define split_gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define split_gpuStreamCreateWithPriority hipStreamCreateWithPriority
#define split_gpuStreamDefault hipStreamDefault
#define split_gpuEventCreate hipEventCreate
#define split_gpuEventCreateWithFlags hipEventCreateWithFlags
#define split_gpuEventDestroy hipEventDestroy
#define split_gpuEventQuery hipEventQuery
#define split_gpuEventRecord hipEventRecord
#define split_gpuEventSynchronize hipEventSynchronize
#define split_gpuEventElapsedTime hipEventElapsedTime
#define split_gpuError_t hipError_t
#define split_gpuSuccess hipSuccess
#define split_gpuStream_t hipStream_t
#define split_gpuDeviceProp hipDeviceProp_t
#define split_gpuEvent_t hipEvent_t
#define split_gpuEventDefault hipEventDefault
#define split_gpuEventBlockingSync hipEventBlockingSync
#define split_gpuEventDisableTiming hipEventDisableTiming
#define split_gpuMemcpyKind hipMemcpyKind
#define split_gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define split_gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define split_gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define split_gpuMemcpyToSymbol hipMemcpyToSymbol
#define split_gpuCpuDeviceId hipCpuDeviceId
#define split_gpuMemoryAdvise hipMemoryAdvise
#define split_gpuMemAdvise hipMemAdvise

#endif

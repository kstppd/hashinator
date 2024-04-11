/* File:    defaults.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 * Description: A hybrid hashmap that can operate on both
 *              CPUs and GPUs using CUDA unified memory.
 *
 * This program is free software; you can redistribute it and/or
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
#include "hashfunctions.h"

namespace Hashinator {
namespace defaults {
#ifdef __NVCC__
constexpr int WARPSIZE = 32;
constexpr int BUCKET_OVERFLOW = 32;
#elif (__HIP__ && __AMDGCN_WAVEFRONT_SIZE)
// AMDs GPU warp size depends on the GPU architecture
constexpr int WARPSIZE = __AMDGCN_WAVEFRONT_SIZE;
constexpr int BUCKET_OVERFLOW = __AMDGCN_WAVEFRONT_SIZE;
#else
constexpr int BUCKET_OVERFLOW = 32; // to allow cpu only mode
#ifndef HASHINATOR_CPU_ONLY_MODE
#error "Warp size not known, please use a CUDA or HIP compiler."
#endif
#endif
constexpr int elementsPerWarp = 1;
constexpr int MAX_BLOCKSIZE = 1024;
template <typename T>
using DefaultHashFunction = HashFunctions::Fibonacci<T>;
} // namespace defaults

struct Info {
   Info(){};
   Info(int sz)
       : sizePower(sz), fill(0), currentMaxBucketOverflow(defaults::BUCKET_OVERFLOW), tombstoneCounter(0),
         err(status::invalid) {}
   int sizePower;
   size_t fill;
   size_t currentMaxBucketOverflow;
   size_t tombstoneCounter;
   status err;
};

} // namespace Hashinator

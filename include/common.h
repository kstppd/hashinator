/* File:    common.h
 * Authors: Kostis Papadakis (2023)
 *
 * Description: Common tools used in hashinator
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
#ifdef HASHINATOR_CPU_ONLY_MODE
#define HASHINATOR_DEVICEONLY
#define HASHINATOR_HOSTDEVICE
#else
#define HASHINATOR_DEVICEONLY __device__
#define HASHINATOR_HOSTDEVICE __host__ __device__
#endif

/**
 * @brief Computes the next power of 2 greater than or equal to a given value.
 *
 * @param v The value for which to compute the next power of 2.
 * @return The next power of 2 greater than or equal to the input value.
 * Modified from (http://www-graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2) to support 64-bit uints
 */
HASHINATOR_HOSTDEVICE
constexpr inline size_t nextPow2(size_t v) noexcept {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}

/**
 * @brief Computes the next optimal overflow for the hasher kernels
 */
HASHINATOR_HOSTDEVICE
//[[nodiscard]]
constexpr inline size_t nextOverflow(size_t currentOverflow, size_t virtualWarp) noexcept {
    size_t remainder = currentOverflow % virtualWarp;
    return ((remainder)==0)?currentOverflow: currentOverflow + (virtualWarp - remainder);
}

inline bool isDeviceAccessible(void* ptr){
#ifdef __NVCC__
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.type != cudaMemoryType::cudaMemoryTypeManaged &&
        attributes.type != cudaMemoryType::cudaMemoryTypeDevice) {
       return false;
    }
    return true;
#endif

#ifdef __HIP__
    hipPointerAttribute_t attributes;
    hipPointerGetAttributes(&attributes, ptr);
    if (attributes.type != hipMemoryType::hipMemoryTypeManaged &&
        attributes.type != hipMemoryType::hipMemoryTypeDevice) {
       return false;
    }
    return true;
#endif
}

/**
 * @brief Enum for error checking in Hahsinator.
 */
namespace Hashinator {
enum status { success, fail, invalid };

/**
 * @brief Enum for specifying backend target.
 */
enum targets { host, device, automatic };
} // namespace Hashinator

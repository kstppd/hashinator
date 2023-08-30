/* File:    hashfunctions.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 * Description: Defines hashfunctions used by Hashinator
 *
 *
 * This file defines the following classes:
 *    --Hashinator::HashFunctions::Murmur;
 *
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
#include "../common.h"
namespace Hashinator {

namespace HashFunctions {

template <typename T>
struct Fibonacci {
   /**
    * @brief Computes a 32-bit hash value using the Fibonacci hash algorithm.
    *
    * @param key The input key to be hashed.
    * @param sizePower The size power for mixing the key.
    * @return uint32_t The computed hash value.
    */
   [[nodiscard]] HOSTDEVICE inline static constexpr uint32_t fibhash(uint32_t key, const int sizePower) {
      key ^= key >> (32 - sizePower);
      uint32_t retval = (uint64_t)(key * 2654435769ul) >> (32 - sizePower);
      return retval;
   }

   /**
    * @brief Computes a 64-bit hash value using the Fibonacci hash algorithm.
    *
    * @param key The input key to be hashed.
    * @param sizePower The size power for mixing the key.
    * @return uint64_t The computed hash value.
    */
   [[nodiscard]] HOSTDEVICE inline static uint64_t constexpr fibhash(uint64_t key, int sizePower) {
      key ^= key >> (64 - sizePower);
      uint64_t retval = (key * 11400714819323198485ull) >> (64 - sizePower);
      return retval;
   }

   /**
    * @brief Computes a hash value using the Fibonacci hash algorithm.
    *
    * This function is a template wrapper that calls the appropriate fibhash
    * function based on the key type.
    *
    * @param key The input key to be hashed.
    * @param sizePower The size power for mixing the key.
    * @return T The computed hash value.
    */
   [[nodiscard]] HOSTDEVICE inline static constexpr T _hash(T key, const int sizePower) {
      return fibhash(key, sizePower);
   }
};
} // namespace HashFunctions
} // namespace Hashinator

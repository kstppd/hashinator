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

namespace Hashinator{
   namespace defaults{
      constexpr int WARPSIZE = 64;
      constexpr int MAX_BLOCKSIZE = 1024;
      constexpr int BUCKET_OVERFLOW = 64;
      constexpr int elementsPerWarp =  1;
      template <typename T >
      using  DefaultHashFunction=HashFunctions::Murmur<T>;
   } //namespace defaults;
} //namespace Hashinator

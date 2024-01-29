#pragma once
/* File:    unordered_set.h
 * Authors: Kostis Papadakis, Urs Ganse and Markus Battarbee (2023)
 *
 * This file defines the following classes:
 *    --Hashinator::Unordered_Set;
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
#define SPLIT_CPU_ONLY_MODE
#endif
#include "../../common.h"
#include "../../splitvector/gpu_wrappers.h"
#include "../../splitvector/split_allocators.h"
#include "../../splitvector/splitvec.h"
#include "../defaults.h"
#include "../hashfunctions.h"
#include <algorithm>
#include <stdexcept>
#ifndef HASHINATOR_CPU_ONLY_MODE
#include "../../splitvector/split_tools.h"
#include "../hashers.h"
#endif

namespace Hashinator {
template <typename KEY_TYPE, KEY_TYPE EMPTYBUCKET = std::numeric_limits<KEY_TYPE>::max(),
          KEY_TYPE TOMBSTONE = EMPTYBUCKET - 1, class HashFunction = HashFunctions::Fibonacci<KEY_TYPE>>

class Unordered_Set {

   // members
private:
   Unordered_Set* device_set;
   split::SplitVector<KEY_TYPE> buckets;

   // Constructors Destructors and = Operators
   Unordered_Set() {}
   Unordered_Set(uint32_t sizePower) {}
   Unordered_Set(const Unordered_Set& other) {}
   Unordered_Set(Unordered_Set&& other) noexcept {}
   ~Unordered_Set() {}
   Unordered_Set& operator=(const Unordered_Set& other) {}
   Unordered_Set& operator=(Unordered_Set&& other) noexcept {}
};

} // namespace Hashinator

/* File:    hash_pair.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines a custom pair data structure.
 *
 * This file defines the following classes:
 *    --Hashinator::hash_pair;
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
#include <stdlib.h>
#include <type_traits>

namespace Hashinator {

template <typename T, typename U>
struct hash_pair {

   T first;  /**< The first value in the pair. */
   U second; /**< The second value in the pair. */

   /**
    * @brief Default constructor.
    *
    * Initializes both values with their default constructors.
    */
   HASHINATOR_HOSTDEVICE
   hash_pair() : first(T()), second(U()) {}

   /**
    * @brief Constructor with values.
    *
    * Initializes the pair with the provided values.
    *
    * @param f The value for the first element of the pair.
    * @param s The value for the second element of the pair.
    */
   HASHINATOR_HOSTDEVICE
   hash_pair(const T& f, const U& s) : first(f), second(s) {}

   /**
    * @brief Equality operator.
    *
    * Compares two hash_pair objects for equality.
    *
    * @param y The hash_pair to compare with.
    * @return bool Returns true if both pairs are equal, false otherwise.
    */
   HASHINATOR_HOSTDEVICE
   inline bool operator==(const hash_pair& y) const { return first == y.first && second == y.second; }

   /**
    * @brief Inequality operator.
    *
    * Compares two hash_pair objects for inequality.
    *
    * @param y The hash_pair to compare with.
    * @return bool Returns true if pairs are not equal, false otherwise.
    */
   HASHINATOR_HOSTDEVICE
   inline bool operator!=(const hash_pair& y) const { return !(*this == y); }
};

/**
 * @brief Creates a hash_pair object.
 *
 * Convenience function to create and initialize a hash_pair object with the provided values.
 *
 * @tparam T Type of the first value in the pair.
 * @tparam U Type of the second value in the pair.
 * @param x The value for the first element of the pair.
 * @param y The value for the second element of the pair.
 * @return hash_pair<T, U> The created hash_pair object.
 */
template <class T, class U>
HASHINATOR_HOSTDEVICE inline hash_pair<T, U> make_pair(T x, U y) {
   return hash_pair<T, U>(x, y);
}

} // namespace Hashinator

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
#include<stdlib.h>
#include <type_traits>

namespace Hashinator{
   template<typename T, typename U>
   struct hash_pair{
      //Do not compile if T,U are not POD;
      static_assert(std::is_pod<T>::value && std::is_pod<U>::value,"Hash pair does not work for non POD types");
      // members
      T first;
      U second;
      
      //Constructors
      hash_pair():first(T()),second(U()){}
      hash_pair(const T& f,const U& s):first(f),second(s){}
   };

   template <typename T, typename U >
   __host__ __device__
   bool operator ==(const hash_pair<T,U>& lhs, const hash_pair<T,U>&rhs){
      return lhs.first==rhs.first && lhs.second==rhs.second;
   }

   template <typename T, typename U >
   __host__ __device__
   bool operator !=(const hash_pair<T,U>& lhs, const hash_pair<T,U>&rhs){
      return !(lhs==rhs);
   }
}//namespace Hashinator

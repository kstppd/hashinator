/* File:    hash_pair.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines a custom pair data structure
 *              that also stores the overflowing offset.
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
      unsigned int offset; //overflowing offset from ideal position
      
      //Constructors
      hash_pair():first(T()),second(U()),offset(0){}
      hash_pair(const T& f,const U& s):first(f),second(s),offset(0){}
      explicit hash_pair(const T& f,const U& s,unsigned char d):first(f),second(s),offset(d){}
   };
}//namespace Hashinator

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
      // members
      T first;
      U second;
      
      //Constructors
      hash_pair():first(T()),second(U()){}
      hash_pair(const T& f,const U& s):first(f),second(s){}
      
      inline bool operator==(const hash_pair& y)const{
         return first == y.first && second == y.second;
      }

      inline bool operator!=(const hash_pair& y)const{
         return !(*this==y);
      }
   };

   template<class T, class U>
   inline hash_pair<T,U> make_pair(T x, U y){
      return hash_pair<T, U>(x, y); 
   }



}//namespace Hashinator


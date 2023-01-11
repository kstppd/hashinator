/* File:    hash_pair.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines a custom pair data structure
 *              that also stores the overflowing offset.
 *
 * This file defines the following classes:
 *    --Hashinator::hash_pair;
 *
 *
 * (c) Copyright 2012-2023 Apache License version 2.0 or later
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

#pragma once 
#include<stdlib.h>
#include <type_traits>

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


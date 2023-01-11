/* File:    hashfunctions.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: Defines hashfunctions used by Hashinator
 *              
 *
 * This file defines the following classes:
 *    --Hashinator::HashFunctions::Murmur;
 *
 *
 * (c) Copyright 2012-2023 Apache License version 2.0 or later
 * */
#pragma once
namespace Hashinator{

   namespace HashFunctions{

      template<typename T>
      struct Murmur{
         __host__ __device__
         inline static uint32_t _hash(T key){
            key ^= key >> 16;
            key *= 0x85ebca6b;
            key ^= key >> 13;
            key *= 0xc2b2ae35;
            key ^= key >> 16;
            return key;
         }
      };
   }//namespace HashFunctions
}//namespace Hashinator

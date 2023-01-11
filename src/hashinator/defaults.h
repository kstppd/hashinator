/* File:    hasinator.h
 * Authors: Kostis Papadakis and Urs Ganse (2023)
 * Description: A hybrid hashmap that can operate on both 
 *              CPUs and GPUs using CUDA unified memory.
 *
 * (c) Copyright 2012-2023 Apache License version 2.0 or later
 * */
#pragma once

namespace Hashinator{
   namespace defaults{
      constexpr int WARPSIZE = 32;
      constexpr int MAX_BLOCKSIZE = 1024;
   } //namespace defaults;
} //namespace Hashinator

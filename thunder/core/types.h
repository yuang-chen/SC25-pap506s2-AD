#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace thunder {

using uint8_t = unsigned char;
using bmp64_t = unsigned long long int;
using half    = __half;
using half2   = __half2;
using BMP256  = bmp64_t[4];

union Bitmap256_U {
    bmp64_t  bmp64[4];
    uint32_t bmp32[8];
};

enum class BMP_SIZE { BMP64, BMP256 };

struct __align__(16) half8
{
    half2 x, y, z, w;
};

}  // namespace thunder
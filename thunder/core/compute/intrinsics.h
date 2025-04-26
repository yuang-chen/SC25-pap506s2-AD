#pragma once

namespace thunder {

template<typename T>
__device__ __forceinline__ T ld(const T* x)
{
    return __ldg(x);
}

__device__ __forceinline__ int cntbit(long long x)
{
    return __popcll(x);
}

__device__ __forceinline__ int count_bmp256(Bitmap256_U bmp256)
{
    int count = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        count += __popcll(bmp256.bmp32[i]);
    }
    return count;
}

__device__ __forceinline__ int count_bits_by_segment_reverse(const Bitmap256_U& bmp256, int seg, int lid)
{
    auto main = 0;
#pragma unroll
    for (int i = 7; i > seg; i--) {
        main += __popc(bmp256.bmp32[i]);
    }
    auto residual = __popc(bmp256.bmp32[seg] >> lid);
    return main + residual;
}

__device__ __forceinline__ int count_bits_by_segment(const Bitmap256_U& bmp256, int seg, int lid)
{
    auto main = 0;
#pragma unroll
    for (int i = 0; i < seg; i++) {
        main += __popc(bmp256.bmp32[i]);
    }
    auto residual = __popc(bmp256.bmp32[seg] << (32 - lid));
    return main + residual;
}

__device__ __forceinline__ void load_bmp256(BMP256 result, const unsigned long long* bitmaps, int tile_idx)
{
    result[0] = __ldg(&bitmaps[tile_idx * 4]);
    result[1] = __ldg(&bitmaps[tile_idx * 4 + 1]);
    result[2] = __ldg(&bitmaps[tile_idx * 4 + 2]);
    result[3] = __ldg(&bitmaps[tile_idx * 4 + 3]);
}

template<typename T>
__device__ __forceinline__ half as_half(T value)
{
    if constexpr (std::is_same_v<T, half>) {
        return value;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return __float2half_rn(value);
    }
    else {
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "Only half and float types are supported");
        return half{};
    }
}

__device__ __forceinline__ half load_half(const half2* ptr, int idx)
{
    half2 A_val1 = __ldg(ptr + idx / 2);
    return idx % 2 == 0 ? A_val1.x : A_val1.y;
}

}  // namespace thunder
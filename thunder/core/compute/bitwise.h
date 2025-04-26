#pragma once

namespace thunder {

__device__ __forceinline__ void insert_bits(BMP256& result, unsigned bmp32, int group_idx)
{
    auto seg_idx = group_idx / 2;
    auto offset  = group_idx % 2;
    auto bmp64   = result[seg_idx];
    bmp64 &= ~(0xFFFFFFFFULL << (offset * 32));
    bmp64 |= static_cast<bmp64_t>(bmp32) << (offset * 32);
    result[seg_idx] = bmp64;
}

template<typename BitmapType>
inline constexpr size_t Dimension()
{
    static_assert(std::is_same<BitmapType, bmp64_t>::value, "BitmapType must be an bmp64_t.");
    return 8;
};

__device__ __forceinline__ bmp64_t interleave_bits(uint32_t a, uint32_t b)
{
    bmp64_t x = a;
    bmp64_t y = b;

    // Masks for spreading the bits of x and y
    constexpr bmp64_t mask1 = 0x0000FFFF0000FFFFULL;
    constexpr bmp64_t mask2 = 0x00FF00FF00FF00FFULL;
    constexpr bmp64_t mask3 = 0x0F0F0F0F0F0F0F0FULL;
    constexpr bmp64_t mask4 = 0x3333333333333333ULL;
    constexpr bmp64_t mask5 = 0x5555555555555555ULL;

    // Spread the bits of x and y
    x = (x | (x << 16)) & mask1;
    y = (y | (y << 16)) & mask1;

    x = (x | (x << 8)) & mask2;
    y = (y | (y << 8)) & mask2;

    x = (x | (x << 4)) & mask3;
    y = (y | (y << 4)) & mask3;

    x = (x | (x << 2)) & mask4;
    y = (y | (y << 2)) & mask4;

    x = (x | (x << 1)) & mask5;
    y = (y | (y << 1)) & mask5;

    // Interleave the bits of x and y
    return x | (y << 1);
}

}  // namespace thunder
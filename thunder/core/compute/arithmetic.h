#pragma once

namespace thunder {

template<typename T>
__device__ __forceinline__ T zero()
{
    return static_cast<T>(0);
}

template<typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T temp = a;
    a      = b;
    b      = temp;
}

/**
 * @brief Helper to round up to the nearest multiple of 'r'.
 */
constexpr __host__ __device__ __forceinline__ int round_up(int x, int r)
{
    return (x + r - 1) / r * r;
}

/**
 * @brief Dividy x by y and round up.
 */
constexpr __host__ __device__ __forceinline__ int div_up(int x, int y)
{
    return (x + y - 1) / y;
}

/**
 * @brief Compute log base 2 statically. Only works when x
 * is a power of 2 and positive.
 * that this is evaluated statically.
 */
__host__ __device__ __forceinline__ int log2(int x)
{
    if (x >>= 1)
        return log2(x) + 1;
    return 0;
}

/**
 * @brief Find the minimum statically.
 */
template<typename T>
constexpr __host__ __device__ __forceinline__ T min(T a, T b)
{
    return a < b ? a : b;
}

/**
 * @brief Find the maximum statically.
 */
template<typename T>
constexpr __host__ __device__ __forceinline__ T max(T a, T b)
{
    return a > b ? a : b;
}

}  // namespace thunder
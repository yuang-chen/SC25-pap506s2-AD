
#pragma once
#include <cuda_fp16.h>
#include <mma.h>

namespace thunder {

template<typename Frag, typename T>
__device__ __forceinline__ void load_frag(Frag& frag, const T* A_sm, int m, int src_idx);

template<typename Frag>
__device__ __forceinline__ void load_frag(Frag& frag, const float* A_sm, int m, int src_idx)
{
    *reinterpret_cast<float2*>(&frag.x[m * 6]) = *reinterpret_cast<const float2*>(&A_sm[src_idx]);
}

template<typename Frag>
__device__ __forceinline__ void load_frag(Frag& frag, const half* A_sm, int m, int src_idx)
{
    *reinterpret_cast<half2*>(&frag.x[m * 6]) = *reinterpret_cast<const half2*>(&A_sm[src_idx]);
}

template<typename Frag, typename T>
__device__ __forceinline__ void store_frag(T* res, const Frag& frag, int m, int dst_idx);

template<typename Frag>
__device__ __forceinline__ void store_frag(float* res, const Frag& frag, int m, int dst_idx)
{
    *reinterpret_cast<float2*>(&res[dst_idx]) = *reinterpret_cast<const float2*>(&frag.x[m]);
}

template<typename Frag>
__device__ __forceinline__ void store_frag(half* res, const Frag& frag, int m, int dst_idx)
{
    *reinterpret_cast<half2*>(&res[dst_idx]) = *reinterpret_cast<const half2*>(&frag.x[m]);
}

__device__ __forceinline__ void store_float4(half* B_sm, int dst_idx, float4 B_val4)
{
    *reinterpret_cast<float4*>(&B_sm[dst_idx]) = B_val4;
}

}  // namespace thunder
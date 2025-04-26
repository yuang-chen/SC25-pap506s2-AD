#pragma once

#include <thrust/unique.h>

namespace thunder {

__global__ void float_to_half2_kernel(const float* input, half2* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we don't access out of bounds and handle the case where
    // size is odd.
    if (idx * 2 < size) {
        float f1 = input[idx * 2];
        float f2 = (idx * 2 + 1) < size ? input[idx * 2 + 1] : 0.0f;
        // Low 16 bits of the return value correspond to the input a, high 16
        // bits correspond to the input b.

        output[idx] = __floats2half2_rn(f1, f2);
    }
}

// Add this kernel for float to half conversion
__global__ void float_to_half_kernel(const float* input, half* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

template<typename HalfType>
void f2h_vector(const thrust::device_vector<float>& input, thrust::device_vector<HalfType>& output)
{
    static_assert(std::is_same_v<HalfType, half> || std::is_same_v<HalfType, half2>,
                  "HalfType must be either half or half2");

    if constexpr (std::is_same_v<HalfType, half2>) {
        int GridDim  = div_up(output.size(), 256);
        int BlockDim = 256;
        float_to_half2_kernel<<<GridDim, BlockDim>>>(input.data().get(), output.data().get(), input.size());
    }
    else {  // half
        output.resize(input.size());
        int GridDim  = div_up(input.size(), 256);
        int BlockDim = 256;
        float_to_half_kernel<<<GridDim, BlockDim>>>(input.data().get(), output.data().get(), input.size());
    }
}

template<typename HalfType>
void f2h_matrix(const DenseMatrix<float, device_memory>& input, DenseMatrix<HalfType, device_memory>& output)
{
    static_assert(std::is_same_v<HalfType, half> || std::is_same_v<HalfType, half2>,
                  "HalfType must be either half or half2");

    output.resize(input.num_rows, input.num_cols);

    if constexpr (std::is_same_v<HalfType, half2>) {
        ASSERT(input.num_cols % 2 == 0 && "Input matrix columns must be even for half2");
        output.values.resize(input.num_rows * input.num_cols / 2);
    }
    else {
        output.values.resize(input.num_rows * input.num_cols);
    }
    f2h_vector(input.values, output.values);
}

// Kernel for half to float conversion
__global__ void half_to_float_kernel(const half* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

template<typename HalfType>
void h2f_vector(const thrust::device_vector<HalfType>& input, thrust::device_vector<float>& output)
{
    static_assert(std::is_same_v<HalfType, half> || std::is_same_v<HalfType, half2>,
                  "HalfType must be either half or half2");

    if constexpr (std::is_same_v<HalfType, half2>) {
        output.resize(input.size() * 2);
        int GridDim  = div_up(input.size(), 256);
        int BlockDim = 256;
        half2_to_float_kernel<<<GridDim, BlockDim>>>(input.data().get(), output.data().get(), input.size());
    }
    else {  // half
        output.resize(input.size());
        int GridDim  = div_up(input.size(), 256);
        int BlockDim = 256;
        half_to_float_kernel<<<GridDim, BlockDim>>>(input.data().get(), output.data().get(), input.size());
    }
}

template<typename HalfType>
void h2f_matrix(const DenseMatrix<HalfType, device_memory>& input, DenseMatrix<float, device_memory>& output)
{
    static_assert(std::is_same_v<HalfType, half> || std::is_same_v<HalfType, half2>,
                  "HalfType must be either half or half2");

    if constexpr (std::is_same_v<HalfType, half2>) {
        output.resize(input.num_rows, input.num_cols * 2);
    }
    else {
        output.resize(input.num_rows, input.num_cols);
    }

    h2f_vector(input.values, output.values);
}

}  // namespace thunder
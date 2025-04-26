#pragma once

namespace thunder {

template<typename IndexType, typename ValueType>
struct IsSelfLoop {
    __host__ __device__ bool operator()(const thrust::tuple<IndexType, IndexType, ValueType>& edge)
    {
        return thrust::get<0>(edge) == thrust::get<1>(edge);
    }
};

template<typename IndexType, typename ValueType>
struct IsDirectedEdge {
    __host__ __device__ bool operator()(const thrust::tuple<IndexType, IndexType, ValueType>& edge) const
    {
        IndexType row = thrust::get<0>(edge);
        IndexType col = thrust::get<1>(edge);
        return row == col || row > col;
    }
};

template<typename T>
struct IsNonZero {
    __host__ __device__ bool operator()(const T& x)
    {
        return x != T(0);
    }
};

// Functor to check if a value is non-negative
struct IsNonNegative {
    __host__ __device__ bool operator()(int x)
    {
        return x >= 0;
    }
};

template<typename T>
struct is_N {
    T N_;

    __host__ __device__ is_N(T n): N_(n) {}

    __host__ __device__ bool operator()(const T& index) const
    {
        return index == N_;
    }
};

// Functor to check if a value is within a range
struct IsWithRange {
    int min_val, max_val;
    IsWithRange(int min, int max): min_val(min), max_val(max) {}

    __host__ __device__ bool operator()(int x)
    {
        return x >= min_val && x < max_val;
    }
};

}  // namespace thunder

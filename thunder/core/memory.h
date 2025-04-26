#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/omp/execution_policy.h>

namespace thunder {

// Memory space tags
struct host_memory {};
struct device_memory {};

template<typename T, typename MemorySpace>
struct VectorTrait;

template<typename T>
struct VectorTrait<T, host_memory> {
    using MemoryVector = thrust::host_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::host)
    {
        return thrust::host;
    }
};

template<typename T>
struct VectorTrait<T, device_memory> {
    using MemoryVector = thrust::device_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::device)
    {
        return thrust::device;
    }
};

template<typename T, typename MemorySpace>
using VectorType = typename VectorTrait<T, MemorySpace>::MemoryVector;

// Helper type trait to check if T is a thrust::device_vector
template<typename T>
struct is_device_vector: std::false_type {};

template<typename T, typename Alloc>
struct is_device_vector<thrust::device_vector<T, Alloc>>: std::true_type {};

// Function to get the execution policy based on vector type
template<typename Vector>
auto get_exec_policy()
{
    if constexpr (is_device_vector<Vector>::value) {
        return thrust::device;
    }
    else {
        return thrust::omp::par;
    }
}

// Memory allocation policies
// template<typename T, typename MemorySpace>
// struct MemoryPolicy;

// template<typename T>
// struct MemoryPolicy<T, host_memory> {
//     using vector_type = thrust::host_vector<T>;
//     static constexpr auto execution_policy()
//     {
//         return thrust::omp::par;
//     }

//     static vector_type allocate(size_t size)
//     {
//         return vector_type(size);
//     }
// };

// template<typename T>
// struct MemoryPolicy<T, device_memory> {
//     using vector_type = thrust::device_vector<T>;
//     static constexpr auto execution_policy()
//     {
//         return thrust::device;
//     }

//     static vector_type allocate(size_t size)
//     {
//         return vector_type(size);
//     }
// };

// // Helper alias template
// template<typename T, typename MemorySpace>
// using memory_vector = typename MemoryPolicy<T, MemorySpace>::vector_type;

}  // namespace thunder
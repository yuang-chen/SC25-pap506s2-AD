#pragma once

#include <cstdlib>
#include <iostream>

namespace thunder {

#define ASSERT(condition)                                                                                              \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::cerr << "Assertion failed: " << #condition << ", file " << __FILE__ << ", line " << __LINE__          \
                      << std::endl;                                                                                    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

// Error checking helpers
#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error = (call);                                                                                    \
        if (error != cudaSuccess) {                                                                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error)          \
                      << std::endl;                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_CUSPARSE(call)                                                                                           \
    do {                                                                                                               \
        cusparseStatus_t status = (call);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            std::cerr << "CUSPARSE error in " << __FILE__ << ":" << __LINE__ << std::endl;                             \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = (call);                                                                                \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            std::cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl;                               \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)
}  // namespace thunder

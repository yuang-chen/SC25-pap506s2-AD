#include <cusparse.h>
#include <iostream>

// https://github.com/dgSPARSE/dgSPARSE-Lib/blob/main/example/sddmm/sddmm.cu

namespace thunder {

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

// the col-major layout is substantially slower than the row-major one
template<typename IndexType, typename ValueType>
void sddmm_cusparse_csr(CsrMatrix<IndexType, ValueType, device_memory>& A,
                        DenseMatrix<ValueType, device_memory>&          B,
                        DenseMatrix<ValueType, device_memory>&          C)
{
    // Initialize cuSPARSE
    // Host problem definition
    ASSERT(A.num_rows == B.num_rows);  // M
    ASSERT(A.num_cols == C.num_rows);  // N
    ASSERT(B.num_cols == C.num_cols);  // K

    int A_nrow = A.num_rows;
    int A_ncol = A.num_cols;
    int A_nnz  = A.num_entries;
    int B_nrow = B.num_rows;
    int B_ncol = B.num_cols;
    int C_nrow = C.num_rows;
    int C_ncol = C.num_cols;

    int ldb = B_ncol;
    int ldc = C_ncol;

    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int*   dA_offsets = A.row_pointers.data().get();
    int*   dA_columns = A.column_indices.data().get();
    float* dA_values  = A.values.data().get();
    float* dB         = B.values.data().get();
    float* dC         = C.values.data().get();
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matB, matC;
    cusparseSpMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                     A_nrow,
                                     A_ncol,
                                     A_nnz,
                                     dA_offsets,
                                     dA_columns,
                                     dA_values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));
    // Create dense matrix X
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_nrow, B_ncol, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix Y
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C_nrow, C_ncol, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha,
                                            matB,
                                            matC,
                                            &beta,
                                            matA,
                                            CUDA_R_32F,
                                            CUSPARSE_SDDMM_ALG_DEFAULT,
                                            &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha,
                                            matB,
                                            matC,
                                            &beta,
                                            matA,
                                            CUDA_R_32F,
                                            CUSPARSE_SDDMM_ALG_DEFAULT,
                                            dBuffer));
    // execute SpMM
    CHECK_CUSPARSE(cusparseSDDMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha,
                                 matB,
                                 matC,
                                 &beta,
                                 matA,
                                 CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT,
                                 dBuffer));
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer));
}

template<typename IndexType, typename ValueType, typename Config>
void execute_cusparse_sddmm(const Config&                                   config,
                            CsrMatrix<IndexType, ValueType, device_memory>& mat,
                            DenseMatrix<ValueType, device_memory>&          x,
                            DenseMatrix<ValueType, device_memory>&          y)
{
    CUDATimer timer;
    timer.start();
    for (int i = 0; i < config.exec_iterations; i++) {
        sddmm_cusparse_csr(mat, x, y);
    }
    timer.stop();
    auto time = timer.elapsed() / config.exec_iterations;
    printf("[cusparse] csr sddmm time (ms): %8.4f \n", time);

    double gflops = (2.0 * mat.num_entries * x.num_cols) / (time * 1e6);
    printf("[cusparse] gflops: %8.4f\n", gflops);
}

}  // namespace thunder

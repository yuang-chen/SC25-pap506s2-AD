#include <cusparse.h>
#include <iostream>

namespace thunder {

enum class Layout {
    RowMajor,
    ColMajor,
};

template<typename IndexType, typename ValueType>
void spmm_cusparse_csr(CsrMatrix<IndexType, ValueType, device_memory>& mat,
                       DenseMatrix<ValueType, device_memory>&          x,
                       DenseMatrix<ValueType, device_memory>&          y)
{
    const auto A_nrow = mat.num_rows;
    const auto A_ncol = mat.num_cols;
    const auto A_nnz  = mat.num_entries;
    const auto B_nrow = A_ncol;
    const auto B_ncol = x.num_cols;
    const auto C_nrow = A_nrow;
    const auto C_ncol = y.num_cols;
    // ASSERT(x.size() % B_nrow == 0 && x.size() != 0);
    // ASSERT(y.size() % C_nrow == 0 && y.size() != 0);
    ASSERT(B_ncol == C_ncol);

    auto* row_ptr = mat.row_pointers.data().get();
    auto* col_idx = mat.column_indices.data().get();
    auto* values  = mat.values.data().get();
    auto  mat_x   = x.values.data().get();
    auto  mat_y   = y.values.data().get();

    float alpha = 1.0f;
    float beta  = 0.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseOrder_t      order      = CUSPARSE_ORDER_ROW;
    int                  b_ld       = B_ncol;
    int                  c_ld       = C_ncol;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                     A_nrow,
                                     A_ncol,
                                     A_nnz,
                                     (void*)row_ptr,
                                     (void*)col_idx,
                                     (void*)values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_nrow, B_ncol, b_ld, (void*)mat_x, CUDA_R_32F, order));
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C_nrow, C_ncol, c_ld, (void*)mat_y, CUDA_R_32F, order));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           matB,
                                           &beta,
                                           matC,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM

    CHECK_CUSPARSE(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA,
                                matB,
                                &beta,
                                matC,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer));

    // destroy matrix descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

template<typename IndexType, typename ValueType, typename Config>
void execute_cusparse_spmm(const Config&                                   config,
                           CsrMatrix<IndexType, ValueType, device_memory>& mat,
                           DenseMatrix<ValueType, device_memory>&          x,
                           DenseMatrix<ValueType, device_memory>&          y)
{
    CUDATimer timer;
    timer.start();
    for (int i = 0; i < config.exec_iterations; i++) {
        spmm_cusparse_csr(mat, x, y);
    }
    timer.stop();
    auto time = timer.elapsed() / config.exec_iterations;
    printf("[cusparse] csr spmm time (ms): %8.4f\n", time);

    double gflops = (2.0 * mat.num_entries * x.num_cols) / (time * 1e6);
    printf("[cusparse] gflops: %8.4f\n", gflops);
}

}  // namespace thunder

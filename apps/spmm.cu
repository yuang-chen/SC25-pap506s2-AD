#include <thunder.h>

using namespace thunder;

template<typename ShrMat>
auto inline set_threads(const ShrMat& A)
{
    // bmp256
    if constexpr (ShrMat::bmp_size == BMP_SIZE::BMP256) {
        auto grid  = dim3(A.num_rows, 1, 1);
        auto block = dim3(WARP_SIZE, WARPS_BLOCK, 1);
        return std::make_pair(grid, block);
    }
    // bmp64
    else if constexpr (ShrMat::bmp_size == BMP_SIZE::BMP64) {
        auto grid  = dim3(div_up(A.num_rows, 2), 1, 1);
        auto block = dim3(WARP_SIZE, WARPS_BLOCK, 1);
        return std::make_pair(grid, block);
    }
    else {
        printf("Error: unknown matrix type\n");
        std::exit(1);
    }
}

template<typename ShrMat, typename InMat, typename OutMat>
void execute_spmm(Config& config, const ShrMat& A, InMat& B, OutMat& C)
{
    using IndexType        = typename ShrMat::index_type;
    using InType           = typename InMat::value_type;
    using OutType          = typename OutMat::value_type;
    constexpr auto BmpSize = ShrMat::bmp_size;

    static_assert(std::is_same_v<typename ShrMat::value_type, typename InMat::value_type>);
    std::string prefix = describe_ops_types< BmpSize>();

    const auto tile_dim     = ShrMat::bmp_size == BMP_SIZE::BMP256 ? FRAG_DIM : TILE_DIM;
    const auto C_nrow       = A.num_rows * tile_dim;
    const auto C_ncol       = B.num_cols;
    const auto C_nrow_tiles = A.num_rows;
    const auto C_ncol_tiles = div_up(C_ncol, tile_dim);
    //    printf("C_nrow: %d, C_ncol: %d, C_nrow_tiles: %d, C_ncol_tiles: %d\n", C_nrow, C_ncol, C_nrow_tiles,
    //    C_ncol_tiles);
    ASSERT(C_ncol % tile_dim == 0);

    C.resize(C_nrow, C_ncol);
    thrust::fill(C.values.begin(), C.values.end(), 0.0);

    auto A_rowptr_ptr = thrust::raw_pointer_cast(A.row_pointers.data());
    auto A_colidx_ptr = thrust::raw_pointer_cast(A.column_indices.data());
    auto A_offset_ptr = thrust::raw_pointer_cast(A.tile_offsets.data());
    auto A_bitmap_ptr = thrust::raw_pointer_cast(A.bitmaps.data());
    auto A_values_ptr = thrust::raw_pointer_cast(A.values.data());
    auto B_values_ptr = thrust::raw_pointer_cast(B.values.data());
    auto C_values_ptr = thrust::raw_pointer_cast(C.values.data());
    auto A_glbcol_ptr = thrust::raw_pointer_cast(A.column_indices_global.data());

    auto [grid, block] = set_threads(A);

    CUDATimer timer;
    timer.start();
    for (int i = 0; i < config.exec_iterations; i++) {
        if constexpr (BmpSize == BMP_SIZE::BMP64) {
            spmm_shr64<int, bmp64_t, InType, OutType><<<grid, block>>>(A.num_rows,
                                                                       A.num_entries,
                                                                       C_ncol_tiles,
                                                                       A_rowptr_ptr,
                                                                       A_colidx_ptr,
                                                                       A_glbcol_ptr,
                                                                       A_offset_ptr,
                                                                       A_bitmap_ptr,
                                                                       A_values_ptr,
                                                                       B_values_ptr,
                                                                       C_values_ptr);
        }
        else if constexpr (BmpSize == BMP_SIZE::BMP256) {
            spmm_shr256<int, bmp64_t, InType, OutType><<<grid, block>>>(A.num_rows,
                                                                        A.num_entries,
                                                                        C_ncol_tiles,
                                                                        A_rowptr_ptr,
                                                                        A_colidx_ptr,
                                                                        A_glbcol_ptr,
                                                                        A_offset_ptr,
                                                                        A_bitmap_ptr,
                                                                        A_values_ptr,
                                                                        B_values_ptr,
                                                                        C_values_ptr);
        }
        else {
            static_assert(BmpSize == BMP_SIZE::BMP64 || BmpSize == BMP_SIZE::BMP256,
                          "Unsupported BMP type. Only BMP64 and BMP256 are supported.");
        }
    }

    timer.stop();
    auto time = timer.elapsed() / config.exec_iterations;
    printf("%s csr spmm time (ms): %8.4f \n", prefix.c_str(), time);

    double gflops = (2.0 * A.num_entries * B.num_cols) / (time * 1e6);
    printf("%s gflops: %8.4f\n", prefix.c_str(), gflops);
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    CsrMatrix<int, float, device_memory> A_csr;

    Config config = program_options(argc, argv);

    read_matrix_file(A_csr, config.input_file);

    write_matrix_file(A_csr, config.output_file);

    randomize(A_csr.values, {-1.0, 1.0});
    // thrust::sequence(A_csr.values.begin(), A_csr.values.end(), 1.0);

    int A_nrow = A_csr.num_rows;
    int A_ncol = A_csr.num_cols;
    int B_nrow = A_ncol;
    int B_ncol = config.ncol_dense;

    //? Dense matrix B
    DenseMatrix<float, device_memory> B_dense(B_nrow, B_ncol, 1.0);
    randomize(B_dense.values, {-1.0, 1.0});
    // thrust::sequence(B_dense.values.begin(), B_dense.values.end(), 1.0);
    printf("Dense  Matrix B: %d x %d\n", B_nrow, B_ncol);

    //<<<<<<<<<<<<<<
    //<< Ground Truth
    //<<<<<<<<<<<<<<
    printf("\n\n--------------cuSparse----------------\n");
    DenseMatrix<float, device_memory> C_verify(A_nrow, B_ncol, 0.0);
    execute_cusparse_spmm(config, A_csr, B_dense, C_verify);

    // print_mat(C_verify.values, "C_verify [cuSPARSE] ", B_ncol);

    reorder_graph(config, A_csr);

    printf("\n\n-------FULL-------\n");
    {
        ShrunkMatrix<int, float, device_memory> A_shr;
        shrink_columns<16>(A_csr, A_shr);
        BitmapShrunkMatrix<int, half, bmp64_t, 4, device_memory> A_shr_bmp256;
        convert_shr2bmp(A_shr, A_shr_bmp256);

        DenseMatrix<half, device_memory> B_dense_half(B_nrow, B_ncol);
        f2h_matrix(B_dense, B_dense_half);

        DenseMatrix<float, device_memory> C_output;
        execute_spmm(config, A_shr_bmp256, B_dense_half, C_output);

        if (config.verify) {
            report_mismatches(C_verify.values, C_output.values, float(0.1), 32);
        }
    }

    printf("\n\n-------DIA-------\n");
    {
        ShrunkMatrix<int, float, device_memory> A_shr;
        shrink_columns<8>(A_csr, A_shr);
        BitmapShrunkMatrix<int, half, bmp64_t, 1, device_memory> A_shr_bmp64_half;
        convert_shr2bmp(A_shr, A_shr_bmp64_half);
        DenseMatrix<half, device_memory> B_dense_half(B_nrow, B_ncol);
        f2h_matrix(B_dense, B_dense_half);
        DenseMatrix<float, device_memory> C_output;

        execute_spmm(config, A_shr_bmp64_half, B_dense_half, C_output);

        if (config.verify) {
            report_mismatches(C_verify.values, C_output.values, float(0.1), 32);
        }
    }

    return 0;
}
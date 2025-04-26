
#include <thunder.h>

template<typename ShrMat, typename DnMat>
void execute_sddmm(thunder::Config& config, ShrMat& A, DnMat& B, DnMat& C)
{
    using IndexType        = typename ShrMat::index_type;
    using InType           = typename DnMat::value_type;
    using OutType          = typename ShrMat::value_type;
    constexpr auto BmpSize = ShrMat::bmp_size;

    std::string prefix = thunder::describe_ops_types<BmpSize>();

    const auto tile_dim = ShrMat::bmp_size == thunder::BMP_SIZE::BMP256 ? thunder::FRAG_DIM : thunder::TILE_DIM;

    B.resize(thunder::round_up(B.num_rows, tile_dim), thunder::round_up(B.num_cols, tile_dim));
    C.resize(thunder::round_up(C.num_rows, tile_dim), thunder::round_up(C.num_cols, tile_dim));

    const auto B_ncol_tiles = thunder::div_up(B.num_cols, tile_dim);

    auto A_rowptr_ptr = thrust::raw_pointer_cast(A.row_pointers.data());
    auto A_colidx_ptr = thrust::raw_pointer_cast(A.column_indices.data());
    auto A_offset_ptr = thrust::raw_pointer_cast(A.tile_offsets.data());
    auto A_bitmap_ptr = thrust::raw_pointer_cast(A.bitmaps.data());
    auto A_values_ptr = thrust::raw_pointer_cast(A.values.data());
    auto A_glbcol_ptr = thrust::raw_pointer_cast(A.column_indices_global.data());

    auto B_values_ptr = thrust::raw_pointer_cast(B.values.data());
    auto C_values_ptr = thrust::raw_pointer_cast(C.values.data());

    thunder::CUDATimer timer;
    timer.start();
    if constexpr (ShrMat::bmp_size == thunder::BMP_SIZE::BMP256) {
        dim3 grid(A.num_rows, 1, 1);
        dim3 block(thunder::WARP_SIZE, 1, 1);
        for (int i = 0; i < config.exec_iterations; i++) {
            thunder::sddmm_shr256<<<grid, block>>>(A.num_rows,
                                                   A.num_entries,
                                                   B_ncol_tiles,
                                                   A_rowptr_ptr,
                                                   A_colidx_ptr,
                                                   A_glbcol_ptr,
                                                   A_offset_ptr,
                                                   A_bitmap_ptr,
                                                   B_values_ptr,
                                                   C_values_ptr,
                                                   A_values_ptr);
        }
    }
    else if constexpr (ShrMat::bmp_size == thunder::BMP_SIZE::BMP64) {
        dim3 grid(thunder::div_up(A.num_rows, 2), 1, 1);
        dim3 block(thunder::WARP_SIZE, 1, 1);
        for (int i = 0; i < config.exec_iterations; i++) {
            thunder::sddmm_shr64<<<grid, block>>>(A.num_rows,
                                                  A.num_entries,
                                                  B_ncol_tiles,
                                                  A_rowptr_ptr,
                                                  A_colidx_ptr,
                                                  A_glbcol_ptr,
                                                  A_offset_ptr,
                                                  A_bitmap_ptr,
                                                  B_values_ptr,
                                                  C_values_ptr,
                                                  A_values_ptr);
        }
    }
    else {
        static_assert(ShrMat::bmp_size == thunder::BMP_SIZE::BMP256 || ShrMat::bmp_size == thunder::BMP_SIZE::BMP64,
                      "Unsupported BMP type. Only BMP64 and BMP256 are supported.");
    }
    timer.stop();
    auto time = timer.elapsed() / config.exec_iterations;
    printf("%s sddmm time: %8.4f\n", prefix.c_str(), time);

    double gflops = (2.0 * A.num_entries * B.num_cols) / (time * 1e6);
    printf("%s gflops: %8.4f\n", prefix.c_str(), gflops);
}

int main(int argc, char** argv)
{
    thunder::CsrMatrix<int, float, thunder::device_memory> A_csr;

    thunder::Config config = thunder::program_options(argc, argv);

    thunder::read_matrix_file(A_csr, config.input_file);

    thunder::write_matrix_file(A_csr, config.output_file);

    thunder::randomize(A_csr.values, {1.0, 2.0});

    // thrust::sequence(A_csr.values.begin(), A_csr.values.end(), 0.0);

    int A_nrow = A_csr.num_rows;
    int A_ncol = A_csr.num_cols;
    int B_ncol = config.ncol_dense;

    thunder::DenseMatrix<float, thunder::device_memory> B_dense(A_nrow, B_ncol, 1.0);
    thunder::DenseMatrix<float, thunder::device_memory> C_dense(A_ncol, B_ncol, 1.0);
    thunder::randomize(B_dense.values, {-1.0, 1.0});
    // thrust::sequence(B_dense.values.begin(), B_dense.values.end(), 1.0);

    printf("\n--------------cuSparse----------------\n");
    {
        thrust::fill(A_csr.values.begin(), A_csr.values.end(), 0.0);
        thunder::execute_cusparse_sddmm(config, A_csr, B_dense, B_dense);
    }

    reorder_graph(config, A_csr);

    printf("\n--------------FULL----------------\n");
    {
        thunder::ShrunkMatrix<int, float, thunder::device_memory> A_shr;
        thunder::shrink_columns<16>(A_csr, A_shr);
        thunder::BitmapShrunkMatrix<int, float, thunder::bmp64_t, 4, thunder::device_memory> A_shr_bmp256;
        thunder::convert_shr2bmp(A_shr, A_shr_bmp256);
        auto values_verify = A_shr_bmp256.values;

        thunder::DenseMatrix<half, thunder::device_memory> B_dense_half;
        thunder::f2h_matrix(B_dense, B_dense_half);

        execute_sddmm(config, A_shr_bmp256, B_dense_half, B_dense_half);

        if (config.verify) {
            thunder::report_mismatches(values_verify, A_shr_bmp256.values, float(0.1));
        }
    }

    printf("\n--------------DIA----------------\n");
    {
        thunder::ShrunkMatrix<int, float, thunder::device_memory> A_shr;
        thunder::shrink_columns<8>(A_csr, A_shr);
        thunder::BitmapShrunkMatrix<int, float, thunder::bmp64_t, 1, thunder::device_memory> A_shr_bmp64;
        thunder::convert_shr2bmp(A_shr, A_shr_bmp64);
        auto values_verify = A_shr_bmp64.values;

        thunder::DenseMatrix<half, thunder::device_memory> B_dense_half;
        thunder::f2h_matrix(B_dense, B_dense_half);

        execute_sddmm(config, A_shr_bmp64, B_dense_half, B_dense_half);

        if (config.verify) {
            thunder::report_mismatches(values_verify, A_shr_bmp64.values, float(0.1));
        }
    }
}
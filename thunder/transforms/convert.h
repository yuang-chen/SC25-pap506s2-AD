#pragma once

#include <thrust/unique.h>

namespace thunder {

template<typename ShrunkMatrix, typename ShrBmpCSR>
void convert_csr2bmp(ShrunkMatrix mat_input, ShrBmpCSR& mat_output)
{
    using IndexType    = typename ShrBmpCSR::index_type;
    using BitmapType   = typename ShrBmpCSR::bitmap_type;
    using InValueType  = typename ShrunkMatrix::value_type;
    using OutValueType = typename ShrBmpCSR::value_type;

    constexpr auto nbmp64   = ShrBmpCSR::bmp64_count;
    constexpr auto tile_dim = nbmp64 == 1 ? TILE_DIM : FRAG_DIM;
    // Use thrust::device directly for simplicity and readability.

    auto       exec      = thrust::device;
    const auto nnz       = mat_input.num_entries;
    const auto nrow      = mat_input.num_rows;
    const auto ncol      = mat_input.num_cols;
    const auto nrow_tile = div_up(nrow, tile_dim);
    const auto ncol_tile = div_up(ncol, tile_dim);
    ASSERT(nrow_tile * ncol_tile < std::numeric_limits<BitmapType>::max()
           && "BitmapType is not large enough to represent the number of tiles");

    thrust::device_vector<IndexType> row_indices(nnz);
    get_row_indices_from_pointers(mat_input.row_pointers, row_indices);

    thrust::sort_by_key(mat_input.column_indices.begin(),
                        mat_input.column_indices.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_input.values.begin())));
    thrust::stable_sort_by_key(
        row_indices.begin(),
        row_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(mat_input.column_indices.begin(), mat_input.values.begin())));

    thrust::device_vector<BitmapType> tile_indices(nnz);
    thrust::device_vector<BitmapType> pos_in_tile(nnz);

    // Calculate tile indices and pos_in_tiles with a single pass.

    using LocateTile = std::conditional_t<(nbmp64 == 1), LocateTile64<BitmapType>, LocateTile256<BitmapType>>;

    thrust::transform(
        exec,
        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_input.column_indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(), mat_input.column_indices.end())),
        thrust::make_zip_iterator(thrust::make_tuple(tile_indices.begin(), pos_in_tile.begin())),
        LocateTile(ncol_tile));

    // print_vec(tile_indices, "tile_indices: ");
    // print_vec(pos_in_tile, "pos_in_tile: ");

    // Sort based on tile indices. This operation affects the original matrices
    // in-place.
    //! due to this step, we have to utilize a vector of row_indices
    thrust::stable_sort_by_key(
        exec,
        tile_indices.begin(),
        tile_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            row_indices.begin(), mat_input.column_indices.begin(), mat_input.values.begin(), pos_in_tile.begin())));

    // Perform reduction by key in-place where possible.
    // Using Thrust's reduce_by_key to compact and aggregate bitmap
    // pos_in_tiles.
    thrust::device_vector<BitmapType> unique_tile_indices = tile_indices;
    auto tile_indices_end = thrust::unique(exec, unique_tile_indices.begin(), unique_tile_indices.end());
    auto num_tiles        = tile_indices_end - unique_tile_indices.begin();
    unique_tile_indices.erase(tile_indices_end, unique_tile_indices.end());

    thrust::device_vector<BitmapType> bitmaps(num_tiles * nbmp64);
    thrust::device_vector<BitmapType> tile_positions(nnz);

    if constexpr (nbmp64 == 1) {
        auto [unique_tile_indices_end, bitmap_end] = thrust::reduce_by_key(exec,
                                                                           tile_indices.begin(),
                                                                           tile_indices.end(),
                                                                           pos_in_tile.begin(),
                                                                           unique_tile_indices.begin(),
                                                                           bitmaps.begin(),
                                                                           thrust::equal_to<BitmapType>(),
                                                                           thrust::bit_or<BitmapType>());
    }
    else {
        thrust::lower_bound(unique_tile_indices.begin(),
                            unique_tile_indices.end(),
                            tile_indices.begin(),
                            tile_indices.end(),
                            tile_positions.begin());

        thrust::for_each(exec,
                         thrust::make_zip_iterator(thrust::make_tuple(tile_positions.begin(), pos_in_tile.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(tile_positions.end(), pos_in_tile.end())),
                         CombineToBMP256<BitmapType>(thrust::raw_pointer_cast(bitmaps.data())));
    }
    // free vector
    tile_indices.resize(0);
    tile_positions.resize(0);
    pos_in_tile.resize(0);
    tile_indices.shrink_to_fit();
    tile_positions.shrink_to_fit();
    pos_in_tile.shrink_to_fit();

    row_indices.resize(num_tiles);
    row_indices.shrink_to_fit();
    // Setup output matrix dimensions based on TILE_DIM.
    mat_output.resize(nrow_tile, ncol_tile, nnz, num_tiles);

    // Transform tile indices to row and column indices for the output matrix.
    thrust::transform(
        exec,
        unique_tile_indices.begin(),
        unique_tile_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_output.column_indices.begin())),
        COOIndices<IndexType, BitmapType>(ncol_tile));
    // print_vec(mat_output.row_indices, "row_indices: ", 10);
    // print_vec(mat_output.row_indices.end() - 10,
    // mat_output.row_indices.end(),
    //           "row_indices: ", 10);
    // convert row indices to row pointers for CSR
    get_row_pointers_from_indices(mat_output.row_pointers, row_indices);

    // print_vec(row_indices, "row_indices: ");

    // Copying values and computing bmp_offsets is already efficient.
    if constexpr (std::is_same_v<InValueType, OutValueType>) {
        mat_output.values = std::move(mat_input.values);
    }
    else if constexpr (std::is_same_v<InValueType, float> && std::is_same_v<OutValueType, half2>) {
        const auto num_elem = div_up(nnz, 2);
        mat_output.values.resize(num_elem);
        f2h_vector(mat_input.values, mat_output.values);
    }
    else if constexpr (std::is_same_v<InValueType, float> && std::is_same_v<OutValueType, half>) {
        mat_output.values.resize(nnz);
        f2h_vector(mat_input.values, mat_output.values);
    }
    else {
        printf("Unsupported ValueType\n");
        std::exit(1);
    }

    thrust::transform(
        exec, bitmaps.begin(), bitmaps.end(), mat_output.tile_offsets.begin(), BmpPopcount<IndexType, BitmapType>());

    // Convert bit counts to offsets for the bitmap.
    thrust::exclusive_scan(
        exec, mat_output.tile_offsets.begin(), mat_output.tile_offsets.end(), mat_output.tile_offsets.begin(), 0);

    // Copy the final bitmap values to the output matrix.
    // thrust::copy(exec, bitmap.begin(), end_pair.second,
    //              mat_output.values.begin());
    mat_output.bitmaps = std::move(bitmaps);
    // print_vec(mat_output.row_pointers, "row_pointers: ");
    // print_vec(mat_output.column_indices, "column_indices: ");
    // print_vec(mat_output.values, "values: ");
}

template<typename ShrunkMatrix, typename ShrBmpCSR>
void convert_shr2bmp(ShrunkMatrix mat_input, ShrBmpCSR& mat_output)
{
    using IndexType    = typename ShrBmpCSR::index_type;
    using BitmapType   = typename ShrBmpCSR::bitmap_type;
    using InValueType  = typename ShrunkMatrix::value_type;
    using OutValueType = typename ShrBmpCSR::value_type;

    constexpr auto nbmp64   = ShrBmpCSR::bmp64_count;
    constexpr auto tile_dim = nbmp64 == 1 ? TILE_DIM : FRAG_DIM;
    // Use thrust::device directly for simplicity and readability.

    auto       exec      = thrust::device;
    const auto nnz       = mat_input.num_entries;
    const auto nrow      = mat_input.num_rows;
    const auto ncol      = mat_input.num_cols;
    const auto nrow_tile = div_up(nrow, tile_dim);
    const auto ncol_tile = div_up(ncol, tile_dim);
    ASSERT(nrow_tile * ncol_tile < std::numeric_limits<BitmapType>::max()
           && "BitmapType is not large enough to represent the number of tiles");

    thrust::sort_by_key(mat_input.column_indices.begin(),
                        mat_input.column_indices.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(mat_input.row_indices.begin(),
                                                                     mat_input.values.begin(),
                                                                     mat_input.column_indices_global.begin())));
    thrust::stable_sort_by_key(mat_input.row_indices.begin(),
                               mat_input.row_indices.end(),
                               thrust::make_zip_iterator(thrust::make_tuple(mat_input.column_indices.begin(),
                                                                            mat_input.values.begin(),
                                                                            mat_input.column_indices_global.begin())));

    thrust::device_vector<BitmapType> tile_indices(nnz);
    thrust::device_vector<BitmapType> pos_in_tile(nnz);

    // Calculate tile indices and pos_in_tiles with a single pass.

    using LocateTile = std::conditional_t<(nbmp64 == 1), LocateTile64<BitmapType>, LocateTile256<BitmapType>>;

    thrust::transform(
        exec,
        thrust::make_zip_iterator(thrust::make_tuple(mat_input.row_indices.begin(), mat_input.column_indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(mat_input.row_indices.end(), mat_input.column_indices.end())),
        thrust::make_zip_iterator(thrust::make_tuple(tile_indices.begin(), pos_in_tile.begin())),
        LocateTile(ncol_tile));

    // print_vec(tile_indices, "tile_indices: ");
    // print_vec(pos_in_tile, "pos_in_tile: ");

    // Sort based on tile indices. This operation affects the original matrices
    // in-place.
    //! due to this step, we have to utilize a vector of row_indices
    thrust::stable_sort_by_key(exec,
                               tile_indices.begin(),
                               tile_indices.end(),
                               thrust::make_zip_iterator(thrust::make_tuple(mat_input.row_indices.begin(),
                                                                            mat_input.column_indices.begin(),
                                                                            mat_input.values.begin(),
                                                                            pos_in_tile.begin(),
                                                                            mat_input.column_indices_global.begin())));

    // Perform reduction by key in-place where possible.
    // Using Thrust's reduce_by_key to compact and aggregate bitmap
    // pos_in_tiles.
    thrust::device_vector<BitmapType> unique_tile_indices = tile_indices;
    auto tile_indices_end = thrust::unique(exec, unique_tile_indices.begin(), unique_tile_indices.end());
    auto num_tiles        = tile_indices_end - unique_tile_indices.begin();
    unique_tile_indices.erase(tile_indices_end, unique_tile_indices.end());

    thrust::device_vector<BitmapType> bitmaps(num_tiles * nbmp64);
    thrust::device_vector<BitmapType> tile_positions(nnz);

    if constexpr (nbmp64 == 1) {
        auto [unique_tile_indices_end, bitmap_end] = thrust::reduce_by_key(exec,
                                                                           tile_indices.begin(),
                                                                           tile_indices.end(),
                                                                           pos_in_tile.begin(),
                                                                           unique_tile_indices.begin(),
                                                                           bitmaps.begin(),
                                                                           thrust::equal_to<BitmapType>(),
                                                                           thrust::bit_or<BitmapType>());
    }
    else {
        thrust::lower_bound(unique_tile_indices.begin(),
                            unique_tile_indices.end(),
                            tile_indices.begin(),
                            tile_indices.end(),
                            tile_positions.begin());

        thrust::for_each(exec,
                         thrust::make_zip_iterator(thrust::make_tuple(tile_positions.begin(), pos_in_tile.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(tile_positions.end(), pos_in_tile.end())),
                         CombineToBMP256<BitmapType>(thrust::raw_pointer_cast(bitmaps.data())));
    }
    // free vector
    tile_indices.resize(0);
    tile_positions.resize(0);
    pos_in_tile.resize(0);
    tile_indices.shrink_to_fit();
    tile_positions.shrink_to_fit();
    pos_in_tile.shrink_to_fit();

    thrust::device_vector<IndexType> row_indices(num_tiles);
    // Setup output matrix dimensions based on TILE_DIM.
    mat_output.resize(nrow_tile, ncol_tile, nnz, num_tiles);

    // Transform tile indices to row and column indices for the output matrix.
    thrust::transform(
        exec,
        unique_tile_indices.begin(),
        unique_tile_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_output.column_indices.begin())),
        COOIndices<IndexType, BitmapType>(ncol_tile));
    // print_vec(mat_output.row_indices, "row_indices: ", 10);
    // print_vec(mat_output.row_indices.end() - 10,
    // mat_output.row_indices.end(),
    //           "row_indices: ", 10);
    // convert row indices to row pointers for CSR
    get_row_pointers_from_indices(mat_output.row_pointers, row_indices);

    // print_vec(row_indices, "row_indices: ");

    // Copying values and computing bmp_offsets is already efficient.
    if constexpr (std::is_same_v<InValueType, OutValueType>) {
        mat_output.values = std::move(mat_input.values);
    }
    else if constexpr (std::is_same_v<InValueType, float> && std::is_same_v<OutValueType, half2>) {
        const auto num_elem = div_up(nnz, 2);
        mat_output.values.resize(num_elem);
        f2h_vector(mat_input.values, mat_output.values);
    }
    else if constexpr (std::is_same_v<InValueType, float> && std::is_same_v<OutValueType, half>) {
        mat_output.values.resize(nnz);
        f2h_vector(mat_input.values, mat_output.values);
    }
    else {
        printf("Unsupported ValueType\n");
        std::exit(1);
    }

    thrust::transform(
        exec, bitmaps.begin(), bitmaps.end(), mat_output.tile_offsets.begin(), BmpPopcount<IndexType, BitmapType>());

    // Convert bit counts to offsets for the bitmap.
    thrust::exclusive_scan(
        exec, mat_output.tile_offsets.begin(), mat_output.tile_offsets.end(), mat_output.tile_offsets.begin(), 0);

    // Copy the final bitmap values to the output matrix.
    // thrust::copy(exec, bitmap.begin(), end_pair.second,
    //              mat_output.values.begin());
    mat_output.bitmaps               = std::move(bitmaps);
    mat_output.column_indices_global = std::move(mat_input.column_indices_global);
    // print_vec(mat_output.row_pointers, "row_pointers: ");
    // print_vec(mat_output.column_indices, "column_indices: ");
    // print_vec(mat_output.values, "values: ");
}

__global__ void dense2bmp_kernel(const float* input, float* output, int nrow, int ncol)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = ncol * nrow;
    if (idx < size) {
        // Original position
        int row = idx / ncol;
        int col = idx % ncol;

        // Block indices
        int tile_row = row / TILE_DIM;
        int tile_col = col / TILE_DIM;

        // Position within the block
        int row_in_tile = row % TILE_DIM;
        int col_in_tile = col % TILE_DIM;

        // Compute new index based on block and position within block
        int ncol_tile  = div_up(ncol, TILE_DIM);
        int target_idx = (tile_row * ncol_tile + tile_col) * TILE_DIM * TILE_DIM + row_in_tile * TILE_DIM + col_in_tile;

        // Copy element to new position
        if (target_idx < size) {
            output[target_idx] = input[idx];
        }
    }
}

void convert_dense2bmp(const thrust::device_vector<float>& input,
                       thrust::device_vector<float>&       output,
                       int                                 nrow,
                       int                                 ncol)
{
    ASSERT(ncol % TILE_DIM == 0 && "The columns of the dense matrix must be a multiple of [TILE_DIM = 8]");

    int size     = nrow * ncol;
    int blockDim = 256;
    int gridDim  = div_up(size, blockDim);
    output.resize(size);
    dense2bmp_kernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), nrow, ncol);
}

template<typename DenseMatrix>
void convert_dense2bmp(const DenseMatrix& input, DenseMatrix& output)
{
    ASSERT(input.num_rows == output.num_rows && input.num_cols == output.num_cols);
    ASSERT(input.num_cols % TILE_DIM == 0 && "The columns of the dense matrix must be a multiple of [TILE_DIM = 8]");

    const auto nrow = output.num_rows;
    const auto ncol = output.num_cols;

    const int size     = nrow * ncol;
    int       blockDim = 256;
    int       gridDim  = div_up(size, blockDim);
    dense2bmp_kernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(input.values.data()), thrust::raw_pointer_cast(output.values.data()), nrow, ncol);
}

__global__ void bmp2dense_kernel(const float* input, float* output, int nrow, int ncol)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = ncol * nrow;
    if (idx < size) {
        // Compute the number of tiles per column
        int ncol_tile = div_up(ncol, TILE_DIM);

        // Original tile indices
        int tile_idx = idx / (TILE_DIM * TILE_DIM);
        int tile_row = tile_idx / ncol_tile;
        int tile_col = tile_idx % ncol_tile;

        // Position within the tile
        int pos_within_tile = idx % (TILE_DIM * TILE_DIM);
        int row_in_tile     = pos_within_tile / TILE_DIM;
        int col_in_tile     = pos_within_tile % TILE_DIM;

        // Compute the original position
        int row          = tile_row * TILE_DIM + row_in_tile;
        int col          = tile_col * TILE_DIM + col_in_tile;
        int original_idx = row * ncol + col;

        if (original_idx < size) {
            output[original_idx] = input[idx];
        }
    }
}

template<typename ValueType>
void convert_bmp2dense(const DenseMatrix<ValueType, device_memory>& input,
                       DenseMatrix<ValueType, device_memory>&       output)
{
    ASSERT(input.num_rows == output.num_rows && input.num_cols == output.num_cols);
    ASSERT(input.num_cols % TILE_DIM == 0 && "The columns of the dense matrix must be a multiple of [TILE_DIM = 8]");

    const auto nrow = output.num_rows;
    const auto ncol = output.num_cols;
    int        size = nrow * ncol;

    int blockDim = 256;
    int gridDim  = div_up(size, blockDim);

    // Launch the kernel
    bmp2dense_kernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(input.values.data()), thrust::raw_pointer_cast(output.values.data()), nrow, ncol);
}

template<typename ValueType>
void transpose_dense_memory_layout(thrust::device_vector<ValueType>& dense_mat_row_major, int nrow, int ncol)
{
    // Device vectors
    thrust::device_vector<float> dense_mat_col_major(nrow * ncol);
    thrust::device_vector<int>   indices(nrow * ncol);

    // Fill indices with sequential values
    thrust::sequence(thrust::device, indices.begin(), indices.end());

    // Transform indices to their new positions in column-major order
    thrust::device_vector<int> new_indices(nrow * ncol);
    thrust::transform(
        thrust::device, indices.begin(), indices.end(), new_indices.begin(), TransposeIndices(nrow, ncol));

    // Scatter the elements to their new positions
    thrust::scatter(thrust::device,
                    dense_mat_row_major.begin(),
                    dense_mat_row_major.end(),
                    new_indices.begin(),
                    dense_mat_col_major.begin());

    std::swap(dense_mat_row_major, dense_mat_col_major);
}

template<typename ValueType>
void transpose_dense_memory_layout(const DenseMatrix<ValueType, device_memory>& input,
                                   DenseMatrix<ValueType, device_memory>&       output)
{
    // Device vectors
    ASSERT(input.num_rows == output.num_rows && input.num_cols == output.num_cols);

    const auto nrow          = input.num_rows;
    const auto ncol          = input.num_cols;
    auto&      val_row_major = input.values;

    auto&                      val_col_major = output.values;
    thrust::device_vector<int> indices(nrow * ncol);

    // Fill indices with sequential values
    thrust::sequence(thrust::device, indices.begin(), indices.end());

    // Transform indices to their new positions in column-major order
    thrust::device_vector<int> new_indices(nrow * ncol);
    thrust::transform(
        thrust::device, indices.begin(), indices.end(), new_indices.begin(), TransposeIndices(nrow, ncol));

    // Scatter the elements to their new positions
    thrust::scatter(
        thrust::device, val_row_major.begin(), val_row_major.end(), new_indices.begin(), val_col_major.begin());

    // std::swap(val_row_major, val_col_major);
}

}  // namespace thunder
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <unordered_map>

namespace thunder {

template<typename IndexType>
std::unordered_map<IndexType, IndexType> inplace_deduplication(thrust::host_vector<IndexType>& array)
{
    const auto length = array.size();
    ASSERT(length > 0);
    int                                      loc = 0, cur = 1;
    std::unordered_map<IndexType, IndexType> g2l;  // global to local
    g2l[array[0]] = 0;
    while (cur < length) {
        if (array[cur] != array[cur - 1]) {
            loc++;
            array[loc] = array[cur];
            // mapping from j to TC_block column index.[]
            g2l[array[cur]] = loc;
        }
        cur++;
    }
    // for (auto &[key, value] : g2l)
    //     printf("key: %d, value: %d\n", key, value);
    return g2l;
}

template<int TileDim, typename CsrMatrix, typename ShrunkMatrix>
void shrink_columns(const CsrMatrix& csr, ShrunkMatrix& mix)
{
    using ValueType = typename CsrMatrix::value_type;
    using IndexType = typename CsrMatrix::index_type;

    const auto nrow      = csr.num_rows;
    const auto ncol      = csr.num_cols;
    const auto nnz       = csr.num_entries;
    const auto nrow_tile = div_up(nrow, TileDim);
    // input tensors.
    thrust::host_vector<IndexType> row_idx(nnz);  // really needed?
    thrust::host_vector<IndexType> tile_counts_per_row(nrow_tile);
    thrust::host_vector<IndexType> local_indices(nnz);
    thrust::host_vector<IndexType> global_indices = csr.column_indices;
    thrust::host_vector<IndexType> row_ptr        = csr.row_pointers;
    // thrust::host_vector<ValueType> values = csr.values;

    IndexType total_tile_count = 0;
    IndexType largest_idx      = 0;
    // csr --> coo
#pragma omp parallel for
    for (IndexType i = 0; i < nrow; i++) {
        for (IndexType j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            row_idx[j] = i;
    }

#pragma omp parallel for reduction(+ : total_tile_count) reduction(max : largest_idx)
    for (IndexType i = 0; i < nrow + 1; i += TileDim) {
        IndexType tile_row      = i / TileDim;
        IndexType row_ptr_start = row_ptr[i];
        IndexType row_ptr_end   = row_ptr[min(i + TileDim, nrow)];
        IndexType nnz_window    = row_ptr_end - row_ptr_start;
        if (nnz_window == 0) {
            continue;
        }

        thrust::host_vector<IndexType> window_indices(nnz_window);
        thrust::copy(
            global_indices.begin() + row_ptr_start, global_indices.begin() + row_ptr_end, window_indices.begin());

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(window_indices.begin(), window_indices.begin() + nnz_window);

        // Step-2: Deduplication of the edge id array.
        //! the column indices are unique in the window, and then mapped with
        //! the counts (new index ids).
        // e.g., unique colidx <key>: 1 2 4 5 8, mapped with <value>: 0 1 2 3 4
        auto unique_g2l_map = inplace_deduplication(window_indices);

        // generate meta information for the tiles
        const IndexType unique_col_idx_count = unique_g2l_map.size();
        largest_idx                          = std::max(largest_idx, unique_col_idx_count - 1);
        tile_counts_per_row[tile_row]        = div_up(unique_col_idx_count, TileDim);
        total_tile_count += tile_counts_per_row[tile_row];

        // relabel the column indices.
        for (auto j = row_ptr_start; j < row_ptr_end; j++) {
            auto global_idx  = global_indices[j];
            local_indices[j] = unique_g2l_map[global_idx];
        }
    }

    mix.num_rows              = nrow;
    mix.num_cols              = ncol;
    mix.num_entries           = nnz;
    mix.num_tiles             = total_tile_count;
    mix.row_indices           = row_idx;
    mix.values                = csr.values;
    mix.column_indices_global = global_indices;
    mix.tile_counts_per_row   = tile_counts_per_row;
    mix.column_indices        = local_indices;
}

template<typename CsrMatrix, typename ShrunkMatrix>
void shrink_columns_gpu(CsrMatrix& csr, ShrunkMatrix& mix)
{
    using ValueType = typename CsrMatrix::value_type;
    using IndexType = typename CsrMatrix::index_type;

    const auto nrow      = csr.num_rows;
    const auto ncol      = csr.num_cols;
    const auto nnz       = csr.num_entries;
    const auto nrow_tile = div_up(nrow, TILE_DIM);
    // input tensors.
    thrust::device_vector<IndexType> row_idx(nnz);
    thrust::device_vector<IndexType> tile_counts_per_row(nrow_tile);
    thrust::device_vector<IndexType> local_indices(nnz);
    const auto&                      col_idx = csr.column_indices;
    const auto&                      row_ptr = csr.row_pointers;

    IndexType total_tile_count = 0;

    // csr --> coo
    get_row_indices_from_pointers(row_ptr, row_idx);

    std::vector<cudaStream_t> streams(nrow_tile);
    for (int i = 0; i < nrow_tile; i++) {
        cudaStreamCreate(&streams[i]);
    }

// #pragma omp parallel for reduction(+ : total_tile_count)
#pragma unroll
    for (IndexType i = 0; i < nrow + 1; i += TILE_DIM) {
        auto& stream = streams[i / TILE_DIM];
        auto  exec   = thrust::device;  // thrust::cuda::par.on(stream);

        IndexType tile_row      = i / TILE_DIM;
        IndexType row_ptr_start = row_ptr[i];
        IndexType row_ptr_end   = row_ptr[min(i + TILE_DIM, nrow)];
        IndexType nnz_window    = row_ptr_end - row_ptr_start;
        if (nnz_window) {
            continue;
        }

        thrust::device_vector<IndexType> local_colidx(col_idx.begin() + row_ptr_start, col_idx.begin() + row_ptr_end);
        // thrust::copy(col_idx.begin() + row_ptr_start,
        //              col_idx.begin() + row_ptr_end, local_colidx.begin());

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(exec, local_colidx.begin(), local_colidx.begin() + nnz_window);

        // auto unique_g2l_map = inplace_deduplication(window_indices);
        auto new_end = thrust::unique(exec, local_colidx.begin(), local_colidx.begin() + nnz_window);
        local_colidx.erase(new_end, local_colidx.end());

        thrust::device_vector<IndexType> position(local_colidx.size());

        // generate tile_counts_per_row --> number of TC_blcok in each row
        // window.
        tile_counts_per_row[tile_row] = div_up(local_colidx.size(), TILE_DIM);
        total_tile_count += tile_counts_per_row[tile_row];

        // Map original column indices to their new positions within the tensor
        // core blocks
        thrust::device_vector<IndexType> colix_original(col_idx.begin() + row_ptr_start, col_idx.begin() + row_ptr_end);
        thrust::device_vector<IndexType> positions(colix_original.size());

        // Find the new positions of the original indices in the deduplicated
        // array
        thrust::lower_bound(thrust::device,
                            local_colidx.begin(),
                            local_colidx.end(),
                            colix_original.begin(),
                            colix_original.end(),
                            positions.begin());

        // Copy the mapping (i.e., new positions) back to the host vector
        // local_indices
        thrust::copy(exec, positions.begin(), positions.end(), local_indices.begin() + row_ptr_start);
    }

    mix.num_rows              = nrow;
    mix.num_cols              = ncol;
    mix.num_entries           = nnz;
    mix.num_tiles             = total_tile_count;
    mix.row_pointers          = row_ptr;
    mix.row_indices           = row_idx;
    mix.values                = csr.values;
    mix.column_indices_global = col_idx;
    mix.tile_counts_per_row   = std::move(tile_counts_per_row);
    mix.column_indices        = std::move(local_indices);

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", total_tile_count, total_tile_count * 8 * 8);
}

}  // namespace thunder
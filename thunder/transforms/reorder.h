
#pragma once

#include <omp.h>

namespace thunder {

template<typename CsrMatrix, typename Vector>
void build_csr_gpu(CsrMatrix& mat, const Vector& new_id)
{
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;
    ASSERT(mat.num_rows == new_id.size());

    thrust::device_vector<IndexType> new_degree(mat.num_rows, 0);

    // Assign the outdegree to new id using transform
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<IndexType>(0),
                     thrust::make_counting_iterator<IndexType>(mat.num_rows),
                     [new_deg = thrust::raw_pointer_cast(new_degree.data()),
                      row_ptr = thrust::raw_pointer_cast(mat.row_pointers.data()),
                      new_id  = thrust::raw_pointer_cast(new_id.data())] __device__(IndexType i) {
                         new_deg[new_id[i]] = row_ptr[i + 1] - row_ptr[i];
                     });

    // Build new row_index array
    thrust::device_vector<IndexType> new_row(mat.num_rows + 1, 0);
    thrust::inclusive_scan(new_degree.begin(), new_degree.end(), new_row.begin() + 1);

    ASSERT(new_row.back() == mat.num_entries);
    // Allocate memory for new column indices and values
    thrust::device_vector<IndexType> new_col(mat.num_entries);
    thrust::device_vector<ValueType> new_val(mat.num_entries);

    // Build new col_index array and values
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<IndexType>(0),
                     thrust::make_counting_iterator<IndexType>(mat.num_rows),
                     [row_ptr = thrust::raw_pointer_cast(mat.row_pointers.data()),
                      col_idx = thrust::raw_pointer_cast(mat.column_indices.data()),
                      values  = thrust::raw_pointer_cast(mat.values.data()),
                      new_row = thrust::raw_pointer_cast(new_row.data()),
                      new_col = thrust::raw_pointer_cast(new_col.data()),
                      new_val = thrust::raw_pointer_cast(new_val.data()),
                      new_id  = thrust::raw_pointer_cast(new_id.data())] __device__(IndexType i) {
                         IndexType count     = 0;
                         IndexType new_start = new_row[new_id[i]];
                         for (IndexType j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                             new_col[new_start + count] = new_id[col_idx[j]];
                             new_val[new_start + count] = values[j];
                             count++;
                         }
                     });

    // Update the matrix
    mat.row_pointers   = std::move(new_row);
    mat.column_indices = std::move(new_col);
    mat.values         = std::move(new_val);
}

template<typename CsrMatrix, typename Vector>
void sort_by_degree(const CsrMatrix& mat, Vector& new_ids)
{
    ASSERT(mat.num_rows == new_ids.size());
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;
    // Step 1: Compute Out-degrees
    const auto                       num_nodes = mat.num_rows;
    thrust::device_vector<IndexType> outdegree(num_nodes);
    get_row_lengths_from_pointers(outdegree, mat.row_pointers);

    // Step 2: Create a sequence of node IDs
    thrust::device_vector<IndexType> node_id(num_nodes);
    thrust::sequence(node_id.begin(), node_id.end());

    // Step 3: Sort Nodes by Out-degree (Descending)
    // Pair node IDs with their out-degrees and sort by the second element
    // (out-degree)
    thrust::sort_by_key(outdegree.begin(), outdegree.end(), node_id.begin(), thrust::greater<IndexType>());

    thrust::scatter(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(num_nodes),
                    node_id.begin(),
                    new_ids.begin());
}

template<typename Config, typename CsrMatrix>
void reorder_graph(Config config, CsrMatrix& mat)
{
    if (config.reorder == ReorderAlgo::None) {
        return;
    }

    printf("\n\n----------------Reordering Graph----------------\n");
    thrust::device_vector<int> new_ids(mat.num_rows);

    switch (config.reorder) {

        case ReorderAlgo::Sorting: {
            CUDATimer timer;
            timer.start();
            sort_by_degree(mat, new_ids);
            timer.stop();
            printf("[Sorting] Reordering time (ms): %f \n", timer.elapsed());
            break;
        }
        default:
            printf("Invalid reorder algorithm, use original graph\n");
            return;
    }

    CUDATimer timer;
    timer.start();
    build_csr_gpu(mat, new_ids);
    timer.stop();
    printf("[Rebuilding] graph time (ms): %f \n", timer.elapsed());

    // sort the columns per row and remove duplicates
    sort_columns_per_row(mat);
    // verify the result
    check_csr_valid(mat);

    printf("\n");
}

}  // namespace thunder
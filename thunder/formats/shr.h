#pragma once

#include <thrust/sort.h>

namespace thunder {

template<typename IndexType, typename ValueType, typename MemorySpace>
class ShrunkMatrix {
public:
    using index_type = IndexType;
    using value_type = ValueType;

    using IndexVector = VectorType<IndexType, MemorySpace>;
    using ValueVector = VectorType<ValueType, MemorySpace>;

    IndexType num_rows;
    IndexType num_cols;
    IndexType num_tiles;
    IndexType num_entries;

    IndexVector row_indices;
    IndexVector column_indices;
    IndexVector tile_counts_per_row;
    IndexVector column_indices_global;  // the original column indices
    ValueVector values;

    // Default constructor
    ShrunkMatrix() = default;

    // Constructor with dimensions and default value
    ShrunkMatrix(IndexType nrow, IndexType ncol, IndexType nnz, ValueType):
        num_rows(nrow), num_cols(ncol), num_entries(nnz), row_indices(nnz), column_indices(nnz), values(nnz)
    {
    }

    // Resize the matrix
    void resize(IndexType nrow, IndexType ncol, IndexType nnz)
    {
        num_rows    = nrow;
        num_cols    = ncol;
        num_entries = nnz;
        row_indices.resize(num_entries);
        column_indices.resize(num_entries);
        values.resize(num_entries);
    }

    void free()
    {
        num_rows    = 0;
        num_cols    = 0;
        num_tiles   = 0;
        num_entries = 0;

        row_indices.clear();
        row_indices.shrink_to_fit();
        column_indices.clear();
        column_indices.shrink_to_fit();
        tile_counts_per_row.clear();
        tile_counts_per_row.shrink_to_fit();
        column_indices_global.clear();
        column_indices_global.shrink_to_fit();
        values.clear();
        values.shrink_to_fit();
    }
};

}  // namespace thunder
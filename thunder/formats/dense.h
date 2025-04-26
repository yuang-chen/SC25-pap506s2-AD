#pragma once
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

namespace thunder {

template<typename ValueType, typename MemorySpace>
class DenseMatrix {
public:
    using value_type   = ValueType;
    using ValueVector  = VectorType<ValueType, MemorySpace>;
    using memory_space = MemorySpace;
    int num_rows;
    int num_cols;

    ValueVector values;

    // Default constructor
    DenseMatrix() = default;

    DenseMatrix(int nrow, int ncol): num_rows(nrow), num_cols(ncol), values(nrow * ncol) {}

    DenseMatrix(int nrow, int ncol, ValueType val): num_rows(nrow), num_cols(ncol), values(nrow * ncol, val) {}

    // Resize the matrix
    void resize(int nrow, int ncol)
    {
        num_rows = nrow;
        num_cols = ncol;

        values.resize(nrow * ncol);
    }

    // Indexing access
    value_type& operator()(int row, int col)
    {
        return values[row * num_cols + col];
    }

    const value_type& operator()(int row, int col) const
    {
        return values[row * num_cols + col];
    }

    void free()
    {
        num_rows = 0;
        num_cols = 0;
        values.clear();
        values.shrink_to_fit();
    }
};

}  // namespace thunder
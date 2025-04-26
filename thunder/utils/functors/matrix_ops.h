#pragma once

namespace thunder {

struct ComputeOutDegree {
    const int* row_ptr;

    explicit ComputeOutDegree(const int* _row_ptr): row_ptr(_row_ptr) {}

    __host__ __device__ int operator()(int node) const
    {
        return row_ptr[node + 1] - row_ptr[node];
    }
};

// Functor for filling row_indices from csr_input.row_pointers
template<typename IndexType>
struct FillRowIndices {
    const IndexType* row_pointers;
    IndexType*       row_indices;

    explicit FillRowIndices(const IndexType* _row_pointers, IndexType* _row_indices):
        row_pointers(_row_pointers), row_indices(_row_indices)
    {
    }

    __host__ __device__ void operator()(const IndexType row) const
    {
        for (IndexType i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
            row_indices[i] = row;
        }
    }
};

// Struct for calculating the offset based on tile index.
template<typename IndexType, typename BitmapType>
struct COOIndices {
    const BitmapType ncols;
    explicit COOIndices(BitmapType num_cols): ncols(num_cols) {}

    __host__ __device__ thrust::tuple<IndexType, IndexType> operator()(BitmapType tileIndex) const
    {
        IndexType row = static_cast<IndexType>(tileIndex / ncols);
        IndexType col = static_cast<IndexType>(tileIndex % ncols);
        return thrust::make_tuple(row, col);
    }
};

// Functor to compute squared difference from mean
template<typename T>
struct SquaredDiff {
    T mean;
    SquaredDiff(T _mean): mean(_mean) {}
    __host__ __device__ T operator()(const T& x) const
    {
        return (x - mean) * (x - mean);
    }
};

struct TransposeIndices {
    int nrow, ncol;

    explicit TransposeIndices(int _nrow, int _ncol): nrow(_nrow), ncol(_ncol) {}

    __host__ __device__ int operator()(const int& idx) const
    {
        int i = idx / ncol;   // Row index in row-major
        int j = idx % ncol;   // Column index in row-major
        return j * nrow + i;  // New index in column-major
    }
};

// Functor to check if column indices are in ascending order within each row
struct IsColOrder {
    const int* rowptr;
    const int* colidx;

    IsColOrder(const int* rp, const int* ci): rowptr(rp), colidx(ci) {}

    __host__ __device__ bool operator()(int row)
    {
        for (int i = rowptr[row] + 1; i < rowptr[row + 1]; ++i) {
            if (colidx[i] <= colidx[i - 1]) {
                printf("col[%d] %d, col[%d] %d\n", i - 1, colidx[i - 1], i, colidx[i]);
                return false;
            }
        }
        return true;
    }
};

}  // namespace thunder
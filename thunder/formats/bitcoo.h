#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

// #include <traits.h>

namespace thunder {

template<typename IndexType, typename ValueType, typename BitmapType, typename MemorySpace>
class BitmapCOO {
public:
    using index_type   = IndexType;
    using value_type   = ValueType;
    using bitmap_type  = BitmapType;
    using memory_space = MemorySpace;
    using IndexVector  = VectorType<IndexType, MemorySpace>;
    using ValueVector  = VectorType<ValueType, MemorySpace>;
    using BitmapVector = VectorType<BitmapType, MemorySpace>;

    IndexType num_rows{0};
    IndexType num_cols{0};
    IndexType num_tiles{0};
    IndexType num_entries{0};

    IndexVector  row_indices;     // num of tiles
    IndexVector  column_indices;  // num of tiles
    IndexVector  tile_offsets;    // num of tiles+1,  nnz/tile & exclusive_scan
    BitmapVector bitmaps;         // num of tiles
    ValueVector  values;          // num_entries = nnz_COO

    // Default constructor
    BitmapCOO() = default;

    // Constructor with dimensions and default value
    BitmapCOO(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile):
        num_rows(nrow),
        num_cols(ncol),
        num_entries(nnz),
        num_tiles(ntile),
        row_indices(ntile),
        column_indices(ntile),
        tile_offsets(ntile),
        bitmaps(ntile),
        values(nnz)
    {
    }

    // Resize the matrix 256 x 256 (1000) -> 256/8 x 256/8 = 32 x 32 (1000) (16
    // 8x8 tiles)
    void resize(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        num_rows    = nrow;
        num_cols    = ncol;
        num_entries = nnz;
        num_tiles   = ntile;

        row_indices.resize(ntile);
        column_indices.resize(ntile);
        tile_offsets.resize(ntile + 1);
        bitmaps.resize(ntile);
        values.resize(nnz);
    }
};

template<typename IndexType, typename ValueType, typename BitmapType, typename MemorySpace>
class ShrBmpCOO: public BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace> {
public:
    VectorType<IndexType, MemorySpace> row_indices;
    VectorType<IndexType, MemorySpace> column_indices_global;  // size nnz

    ShrBmpCOO(): BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace>() {};

    void resize(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_indices.resize(ntile);
        this->row_indices.resize(ntile);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile + 1);
        this->bitmaps.resize(ntile);
        this->values.resize(nnz);
    }
};

template<typename IndexType, typename ValueType, typename BitmapType, typename MemorySpace>
class HugeBmpCOO: public BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace> {
public:
    HugeBmpCOO(): BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace>() {};

    HugeBmpCOO(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile):
        BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace>(nrow, ncol, nnz, ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_indices.resize(ntile);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * NUM_BMP64 + 1);
        this->bitmaps.resize(ntile * NUM_BMP64);
        this->values.resize(nnz);
    }

    // Resize the matrix 256 x 256 (1000) -> 256/8 x 256/8 = 32 x 32 (1000) (16
    // 8x8 tiles)
    void resize(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_indices.resize(ntile);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * NUM_BMP64 + 1);
        this->bitmaps.resize(ntile * NUM_BMP64);
        this->values.resize(nnz);
    }
};

template<typename IndexType, typename ValueType, typename BitmapType, typename MemorySpace>
class HugeShrBmpCOO: public BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace> {
public:
    VectorType<IndexType, MemorySpace> row_indices;
    VectorType<IndexType, MemorySpace> column_indices_global;  // size nnz

    HugeShrBmpCOO(): BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace>() {};

    HugeShrBmpCOO(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile):
        BitmapCOO<IndexType, ValueType, BitmapType, MemorySpace>(nrow, ncol, nnz, ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_indices.resize(ntile);
        this->row_indices.resize(ntile);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * NUM_BMP64 + 1);
        this->bitmaps.resize(ntile * NUM_BMP64);
        this->values.resize(nnz);
    }

    void resize(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_indices.resize(ntile);
        this->row_indices.resize(ntile);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * NUM_BMP64 + 1);
        this->bitmaps.resize(ntile * NUM_BMP64);
        this->values.resize(nnz);
    }
};

}  // namespace thunder
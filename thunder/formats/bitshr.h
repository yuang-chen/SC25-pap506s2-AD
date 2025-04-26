#pragma once
#include <thrust/sort.h>

namespace thunder {

template<typename IndexType, typename ValueType, typename BitmapType, IndexType Bmp64PerTile, typename MemorySpace>
class BitmapShrunkMatrix {
public:
    using index_type   = IndexType;
    using value_type   = ValueType;
    using bitmap_type  = BitmapType;
    using memory_space = MemorySpace;

    using IndexVector  = VectorType<IndexType, MemorySpace>;
    using ValueVector  = VectorType<ValueType, MemorySpace>;
    using BitmapVector = VectorType<BitmapType, MemorySpace>;

    static constexpr IndexType bmp64_count = Bmp64PerTile;
    static constexpr BMP_SIZE  bmp_size    = bmp64_count == 1 ? BMP_SIZE::BMP64 : BMP_SIZE::BMP256;

    IndexType num_rows{0};
    IndexType num_cols{0};
    IndexType num_tiles{0};
    IndexType num_entries{0};

    IndexVector  row_pointers;           // tile row pointers
    IndexVector  column_indices;         // tile column indices
    IndexVector  tile_offsets;           // nnz offsets for bmp64
    IndexVector  column_indices_global;  // the original column indices
    ValueVector  values;                 // values of nonzeros
    BitmapVector bitmaps;                // num of tiles

    BitmapShrunkMatrix() = default;

    BitmapShrunkMatrix(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_pointers.resize(nrow + 1);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * Bmp64PerTile + 1);
        this->bitmaps.resize(ntile * Bmp64PerTile);
        this->values.resize(nnz);
        this->column_indices_global.resize(nnz);
    }

    void resize(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile)
    {
        this->num_rows    = nrow;
        this->num_cols    = ncol;
        this->num_entries = nnz;
        this->num_tiles   = ntile;

        this->row_pointers.resize(nrow + 1);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile * Bmp64PerTile + 1);
        this->bitmaps.resize(ntile * Bmp64PerTile);
        this->values.resize(nnz);
        this->column_indices_global.resize(nnz);
    }
};

}  // namespace thunder

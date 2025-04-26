#pragma once

namespace thunder {

template<typename T, typename BitmapType>
struct Binarize {
    const T*     input;
    const size_t nrow;
    const size_t ncol;
    const size_t new_ncol;

    Binarize(const T* _input, const size_t _nrow, const size_t _ncol):
        input(_input), nrow(_nrow), ncol(_ncol), new_ncol((ncol - 1) / TILE_DIM + 1)
    {
    }

    __host__ __device__ BitmapType operator()(const size_t& index) const
    {
        size_t     blockX = (index % new_ncol) * TILE_DIM;
        size_t     blockY = (index / new_ncol) * TILE_DIM;
        BitmapType result = 0;

        for (size_t dy = 0; dy < TILE_DIM; ++dy) {
            for (size_t dx = 0; dx < TILE_DIM; ++dx) {
                size_t x = blockX + dx;
                size_t y = blockY + dy;
                if (x < ncol && y < nrow && input[y * ncol + x] > T(0)) {
                    result |= (BitmapType(1) << (dy * TILE_DIM + dx));
                }
            }
        }
        return result;
    }
};

// Struct for locating the tile index and position within a tile.
template<typename BitmapType>
struct LocateTile64 {
    const BitmapType ncols;
    explicit LocateTile64(BitmapType ncol_tile): ncols(ncol_tile) {}

    __host__ __device__ thrust::tuple<BitmapType, BitmapType>
                        operator()(const thrust::tuple<BitmapType, BitmapType>& indices) const
    {
        const BitmapType row        = thrust::get<0>(indices);
        const BitmapType col        = thrust::get<1>(indices);
        BitmapType       tile_index = (row / TILE_DIM) * ncols + (col / TILE_DIM);
        BitmapType       position   = (BitmapType)1 << ((row % TILE_DIM) * TILE_DIM + (col % TILE_DIM));

        return thrust::make_tuple(tile_index, position);
    }
};

// Struct for locating the tile index and position within a tile.
template<typename BitmapType>
struct LocateTile256 {
    const BitmapType ncols;
    explicit LocateTile256(BitmapType num_cols): ncols(num_cols) {}

    __host__ __device__ thrust::tuple<BitmapType, BitmapType>
                        operator()(const thrust::tuple<BitmapType, BitmapType>& indices) const
    {
        const BitmapType row        = thrust::get<0>(indices);
        const BitmapType col        = thrust::get<1>(indices);
        BitmapType       tile_index = (row / FRAG_DIM) * ncols + (col / FRAG_DIM);
        BitmapType       position   = (row % FRAG_DIM) * FRAG_DIM + (col % FRAG_DIM);

        return thrust::make_tuple(tile_index, position);
    }
};

template<typename BitmapType>
struct CombineToBMP256 {
    BitmapType* bitmaps;
    explicit CombineToBMP256(BitmapType* bitmaps_): bitmaps(bitmaps_) {}

    __host__ __device__ void operator()(const thrust::tuple<BitmapType, BitmapType>& tup)
    {
        auto tile_idx   = thrust::get<0>(tup);
        auto pos        = thrust::get<1>(tup);
        auto seg_idx    = pos / 64;
        auto bit_idx    = pos % 64;
        auto global_idx = tile_idx * 4 + seg_idx;
        atomicOr((unsigned long long int*)&bitmaps[global_idx], 1ULL << bit_idx);
    }
};

template<typename IndexType, typename ValueType, typename BitmapType>
struct FindTile {
    const size_t mat_cols;
    const size_t ncols;
    FindTile(size_t num_cols): mat_cols(num_cols), ncols((num_cols - 1) / TILE_DIM + 1) {}

    __host__ __device__ thrust::tuple<BitmapType, BitmapType> operator()(const IndexType& index) const
    {
        const IndexType row        = index / mat_cols;
        const IndexType col        = index % mat_cols;
        BitmapType      tile_index = (row / TILE_DIM) * ncols + col / TILE_DIM;
        BitmapType      position   = static_cast<BitmapType>(1) << ((row % TILE_DIM) * TILE_DIM + col % TILE_DIM);

        return thrust::make_tuple(tile_index, position);
    }
};

template<typename IndexType, typename BitmapType>
struct BmpPopcount {
    __device__ IndexType operator()(BitmapType bmp64)
    {
        return (IndexType)__popcll(bmp64);
    }
};

// Functor to calculate popcount for four 64-bit integers at a time
template<typename IndexType, typename BitmapType>
struct CountBitsPerTile {
    const BitmapType* bitmaps;
    IndexType*        counts;

    CountBitsPerTile(BitmapType* bitmaps_, IndexType* counts_): bitmaps(bitmaps_), counts(counts_) {}

    __host__ __device__ IndexType operator()(IndexType tile_idx)
    {
        // Calculate the starting index for this block in the bitmap storage
        IndexType start_index = tile_idx * 4;
        IndexType count       = __popcll(bitmaps[start_index]) + __popcll(bitmaps[start_index + 1])
                          + __popcll(bitmaps[start_index + 2]) + __popcll(bitmaps[start_index + 3]);

        // Write the computed count directly to the tile offsets
        // If it's not the first block, add the previous offset to form a
        // cumulative sum
        counts[tile_idx] = count;
    }
};

template<typename ValueType>
struct GatherTile {
    const size_t     ncol_tile;
    const ValueType* input;

    GatherTile(size_t _ncol_tile, const ValueType* _input): ncol_tile(_ncol_tile), input(_input) {}
    __host__ __device__ ValueType operator()(const size_t& index) const
    {
        const auto tile_idx    = index >> 6;
        const auto idx_in_tile = index & 63;
        const auto tile_row    = tile_idx / ncol_tile;
        const auto tile_col    = tile_idx % ncol_tile;
        const auto row_in_tile = idx_in_tile >> 3;
        const auto col_in_tile = idx_in_tile & 7;
        const auto idx = (tile_row * ncol_tile << 6) + (row_in_tile * ncol_tile << 3) + (tile_col << 3) + col_in_tile;
        // const auto tile_idx = index / TILE_SIZE;
        // const auto idx_in_tile = index % TILE_SIZE;
        // const auto tile_row = tile_idx / ncol_tile;
        // const auto tile_col = tile_idx % ncol_tile;
        // const auto row_in_tile = idx_in_tile / TILE_DIM;
        // const auto col_in_tile = idx_in_tile % TILE_DIM;
        // const auto idx = tile_row * ncol_tile * TILE_SIZE +
        //                  row_in_tile * ncol_tile * TILE_DIM +
        //                  tile_col * TILE_DIM + col_in_tile;
        return input[idx];
    }
};

template<typename IndexType>
struct GetTileIndex {
    const size_t ncol_tile;

    GetTileIndex(size_t _ncol_tile): ncol_tile(_ncol_tile) {}
    // TODO: this can be accelerated by bitwise operations
    __host__ __device__ IndexType operator()(const IndexType& index) const
    {
        const auto tile_idx    = index >> 6;
        const auto idx_in_tile = index & 63;
        const auto tile_row    = tile_idx / ncol_tile;
        const auto tile_col    = tile_idx % ncol_tile;
        const auto row_in_tile = idx_in_tile >> 3;
        const auto col_in_tile = idx_in_tile & 7;
        const auto idx = (tile_row * ncol_tile << 6) + (row_in_tile * ncol_tile << 3) + (tile_col << 3) + col_in_tile;
        // const auto tile_idx = index / TILE_SIZE;
        // const auto idx_in_tile = index % TILE_SIZE;
        // const auto tile_row = tile_idx / ncol_tile;
        // const auto tile_col = tile_idx % ncol_tile;
        // const auto row_in_tile = idx_in_tile / TILE_DIM;
        // const auto col_in_tile = idx_in_tile % TILE_DIM;
        // const auto idx = tile_row * ncol_tile * TILE_SIZE +
        //                  row_in_tile * ncol_tile * TILE_DIM +
        //                  tile_col * TILE_DIM + col_in_tile;
        return idx;
    }
};

template<typename ValueType>
struct ScatterTile {
    const size_t     ncol_tile;
    const ValueType* input;

    ScatterTile(size_t _ncol_tile, const ValueType* _input): ncol_tile(_ncol_tile), input(_input) {}
    // TODO: this can be accelerated by bitwise operations
    __host__ __device__ ValueType operator()(const size_t& index) const
    {
        const auto ncol_mat = ncol_tile << 3;
        const auto row_idx  = index / ncol_mat;
        const auto col_idx  = index % ncol_mat;
        const auto tile_row = row_idx >> 3;
        const auto tile_col = col_idx >> 3;
        const auto idx      = ((tile_row * ncol_tile + tile_col) << 6) + ((row_idx & 7) << 3) + (col_idx & 7);
        // const auto ncol_mat = ncol_tile * TILE_DIM;
        // const auto row_idx = index / ncol_mat;
        // const auto col_idx = index % ncol_mat;
        // const auto tile_row = row_idx / TILE_DIM;
        // const auto tile_col = col_idx / TILE_DIM;
        // const auto idx = (tile_row * ncol_tile + tile_col) * TILE_SIZE +
        //                  (row_idx % TILE_DIM) * TILE_DIM + (col_idx %
        //                  TILE_DIM);
        return input[idx];
    }
};

template<typename ValueType>
struct Vec2Tile {
    const ValueType* input;

    Vec2Tile(const ValueType* _input): input(_input) {}
    __host__ __device__ ValueType operator()(const size_t& index) const
    {
        const auto tile_idx    = index / TILE_SIZE;
        const auto idx_in_tile = index % TILE_SIZE;
        const auto row_in_tile = idx_in_tile % TILE_DIM;
        const auto col_in_tile = idx_in_tile / TILE_DIM;

        const auto idx = tile_idx * TILE_SIZE + row_in_tile * TILE_DIM + col_in_tile;
        return input[idx];
    }
};

template<typename IndexType>
struct Vec2TileIndex {
    Vec2TileIndex() {}

    __host__ __device__ IndexType operator()(const size_t& index) const
    {
        const auto tile_idx    = index / TILE_SIZE;
        const auto idx_in_tile = index % TILE_SIZE;
        const auto row_in_tile = idx_in_tile % TILE_DIM;
        const auto col_in_tile = idx_in_tile / TILE_DIM;

        const auto idx = tile_idx * TILE_SIZE + row_in_tile * TILE_DIM + col_in_tile;
        return idx;
    }
};

struct PopCount {
    __device__ int operator()(bmp64_t x) const
    {
        return __popcll(x);
    }
};

struct BmpCountWithinRange {
    int min;
    int max;
    explicit BmpCountWithinRange(int min_, int max_): min(min_), max(max_) {}
    __device__ bool operator()(bmp64_t x) const
    {
        return (__popcll(x) > min && __popcll(x) <= max);
    }
};

}  // namespace thunder

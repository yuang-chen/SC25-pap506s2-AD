#pragma once

namespace thunder {

constexpr uint32_t BALLOT_MASK    = 0xffffffff;
constexpr int      WARP_SIZE      = 32;
constexpr int      HALF_WARP_SIZE = WARP_SIZE / 2;
constexpr int      WARPS_BLOCK    = 8;                           // warps per block
constexpr int      THREADS_BLOCK  = WARPS_BLOCK * WARP_SIZE;     // threads per block
constexpr int      TILE64S_WARP   = 2;                           // tiles per warp
constexpr int      TILE64S_BLOCK  = WARPS_BLOCK * TILE64S_WARP;  // tiles per block
constexpr int      WMMA_M         = 16;
constexpr int      WMMA_N         = 16;
constexpr int      WMMA_K         = 16;
constexpr int      TILE_DIM       = 8;
constexpr int      TILE_SIZE      = 64;
constexpr int      FRAG_DIM       = 16;
constexpr int      FRAG_SIZE      = 256;
constexpr int      WARP_GAP       = WARP_SIZE / TILE_DIM * FRAG_DIM;
constexpr int      TILE_GAP       = FRAG_SIZE / 2 + TILE_DIM;
constexpr int      TWO            = 2;
constexpr int      NUM_BMP64      = 4;  // one huge bitmap is 4 small bitmaps
constexpr int      VEC_WIDTH      = 2;

__constant__ int A_frag_offsets[4] = {0, 128, 8, 136};
__constant__ int B_frag_offsets[4] = {0, 8, 128, 136};

}  // namespace thunder
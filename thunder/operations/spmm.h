#pragma once
#include <cuda.h>
#include <mma.h>

namespace thunder {

template<typename IndexType, typename BitmapType, typename ValueType>
__global__ void spmm_fff_bmp64(int               C_nrow,
                               int               C_ncol,
                               const IndexType*  A_rowptr,
                               const IndexType*  A_colidx,
                               const IndexType*  A_offset,
                               const BitmapType* A_bitmap,
                               const ValueType*  A_values,
                               const ValueType*  B_values,
                               ValueType*        C_values)
{
    const auto wid    = threadIdx.x / WARP_SIZE;
    const auto lid    = threadIdx.x % WARP_SIZE;
    const auto C_size = C_nrow * C_ncol;
    // Tile ID, not Thread ID
    const auto tid1 = blockIdx.x * TILE64S_BLOCK + wid * TILE64S_WARP;
    const auto tid2 = tid1 + 1;

    const BitmapType lid_x_2     = BitmapType(lid) << 1;
    const BitmapType bitmask1    = BitmapType(1) << lid_x_2;
    const BitmapType bitmask2    = BitmapType(2) << lid_x_2;
    const BitmapType A_reverse1  = 64 - lid_x_2;
    const BitmapType A_reverse2  = 63 - lid_x_2;
    const BitmapType B_trans_4x8 = ((lid & 3) << 4) + (lid >> 2);
    const BitmapType B_pattern1  = BitmapType(1) << B_trans_4x8;
    const BitmapType B_pattern2  = BitmapType(1) << (B_trans_4x8 + 8);

    __shared__ float C_sm[FRAG_SIZE * 2];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM, FRAG_DIM, float> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    BitmapType A_bmp1;
    BitmapType A_bmp2;

    auto C_row1 = tid1 / C_ncol;
    auto C_col1 = tid1 % C_ncol;
    auto C_row2 = tid2 / C_ncol;
    auto C_col2 = tid2 % C_ncol;

    auto A_row_begin1 = C_row1 < C_nrow ? A_rowptr[C_row1] : A_rowptr[C_nrow];
    auto A_row_begin2 = C_row2 < C_nrow ? A_rowptr[C_row2] : A_rowptr[C_nrow];
    auto A_row_end1   = C_row1 < C_nrow ? A_rowptr[C_row1 + 1] : A_rowptr[C_nrow];
    auto A_row_end2   = C_row2 < C_nrow ? A_rowptr[C_row2 + 1] : A_rowptr[C_nrow];

    auto i1 = A_row_begin1, i2 = A_row_begin2;
    bool process1 = i1 < A_row_end1;
    bool process2 = i2 < A_row_end2;

    if (process1 || process2) {
        nvcuda::wmma::fill_fragment(a_frag, 0.0f);
        nvcuda::wmma::fill_fragment(b_frag, 0.0f);
    }
    while (process1 || process2) {
        if (process1) {
            A_bmp1               = A_bitmap[i1];
            const auto num_bits1 = cntbit(A_bmp1 << A_reverse1);
            const auto num_bits2 = cntbit(A_bmp1 << A_reverse2);
            auto       B_idx     = A_colidx[i1] * C_ncol + C_col1;

            a_frag.x[0] = (A_bmp1 & bitmask1) > 0 ? A_values[A_offset[i1] + num_bits1] : zero<half>();
            a_frag.x[1] = (A_bmp1 & bitmask2) > 0 ? A_values[A_offset[i1] + num_bits2] : zero<half>();

            b_frag.x[0] = B_values[B_idx * TILE_SIZE + B_trans_4x8];
            b_frag.x[1] = B_values[B_idx * TILE_SIZE + B_trans_4x8 + 8];
        }
        if (process2) {
            A_bmp2               = A_bitmap[i2];
            auto       B_idx     = A_colidx[i2] * C_ncol + C_col2;
            const auto num_bits1 = cntbit(A_bmp2 << A_reverse1);
            const auto num_bits2 = cntbit(A_bmp2 << A_reverse2);

            a_frag.x[6] = (A_bmp2 & bitmask1) > 0 ? A_values[A_offset[i2] + num_bits1] : zero<half>();
            a_frag.x[7] = (A_bmp2 & bitmask2) > 0 ? A_values[A_offset[i2] + num_bits2] : zero<half>();
            b_frag.x[6] = B_values[B_idx * TILE_SIZE + B_trans_4x8];
            b_frag.x[7] = B_values[B_idx * TILE_SIZE + B_trans_4x8 + 8];
        }

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        ++i1;
        ++i2;
        process1 = i1 < A_row_end1;
        process2 = i2 < A_row_end2;
    }
    nvcuda::wmma::store_matrix_sync(&C_sm[wid * FRAG_SIZE], acc_frag, FRAG_DIM, nvcuda::wmma::mem_row_major);
    __syncwarp();

    float2 C_result1, C_result2;
    int    base_index = lid / TILE_DIM * FRAG_DIM + lid % TILE_DIM + wid * FRAG_SIZE;

    // Simplified C_result1 calculations
    C_result1.x = C_sm[base_index];
    C_result1.y = C_sm[base_index + TILE_SIZE];

    // Simplified C_result2 calculations
    C_result2.x = C_sm[base_index + TILE_GAP];
    C_result2.y = C_sm[base_index + TILE_GAP + TILE_SIZE];
    auto y_idx1 = tid1 * TILE_SIZE + lid;
    auto y_idx2 = tid2 * TILE_SIZE + lid;

    if (tid1 < C_size) {
        C_values[y_idx1]             = C_result1.x;
        C_values[y_idx1 + WARP_SIZE] = C_result1.y;
    }
    if (tid2 < C_size) {
        C_values[y_idx2]             = C_result2.x;
        C_values[y_idx2 + WARP_SIZE] = C_result2.y;
    }
}

template<typename IndexType, typename InputValueType>
__device__ __forceinline__ void decode_bmp256(int                   tid,
                                              int                   threads_block,
                                              const BMP256&         bmp256,
                                              int                   tile_idx,
                                              const IndexType*      A_offset,
                                              const InputValueType* A_values,
                                              const IndexType*      A_glbcol,
                                              half*                 A_sm,
                                              int*                  glbcol_sm)
{
#pragma unroll
    for (int idx = tid; idx < FRAG_SIZE; idx += threads_block) {
        const int     seg_idx  = idx / TILE_SIZE;
        const int     bit_idx  = idx % TILE_SIZE;
        const bmp64_t bmp64    = bmp256[seg_idx];
        const bmp64_t bit_mask = bmp64_t(1) << bit_idx;
        const bool    valid    = bmp64 & bit_mask;

        if (valid) {
            const int bitcnt     = cntbit(bmp64 << (64 - bit_idx));
            const int bmp64_idx  = tile_idx * 4 + seg_idx;
            const int A_idx      = ld(&A_offset[bmp64_idx]) + bitcnt;
            const int local_col  = tid % FRAG_DIM;
            A_sm[idx]            = as_half(ld(&A_values[A_idx]));
            glbcol_sm[local_col] = ld(&A_glbcol[A_idx]);
        }
    }
}
template<typename InputValueType>
__device__ __forceinline__ void load_dense_tile256(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::col_major>& b_frag,
    const int             base_idx,
    const int*            glbcol_sm,
    const InputValueType* B_values,
    const int             curr_col,
    const int             C_ncol_elems,
    const int             C_bound)
{
#pragma unroll
    for (int idx = 0; idx < 4; idx++) {
        int elem_idx = base_idx + A_frag_offsets[idx];
#pragma unroll
        for (int n = 0; n < 2; n++) {
            elem_idx += n * 16;
            const int dense_row_idx = glbcol_sm[elem_idx / FRAG_DIM];

            const int source_idx  = dense_row_idx * C_ncol_elems + curr_col * FRAG_DIM + elem_idx % FRAG_DIM;
            b_frag.x[2 * idx + n] = source_idx < C_bound ? as_half(ld(&B_values[source_idx])) : zero<half>();
        }
    }
}

template<typename IndexType, typename BitmapType, typename InputValueType, typename OutputValueType>
    requires SparseOpType<IndexType, BitmapType, InputValueType, OutputValueType>
__global__ void spmm_shr256(int A_nrow,
                            int A_nnz,
                            int C_ncol,
                            const IndexType* __restrict__ A_rowptr,
                            const IndexType* __restrict__ A_colidx,
                            const IndexType* __restrict__ A_glbcol,
                            const IndexType* __restrict__ A_offset,
                            const BitmapType* __restrict__ A_bitmap,
                            const InputValueType* __restrict__ A_values,
                            const InputValueType* __restrict__ B_values,
                            OutputValueType* C_values)
{
    using OutType = std::conditional_t<std::is_same_v<OutputValueType, half>, half, float>;

    const int bid           = blockIdx.x;               // block id == row id
    const int wid           = threadIdx.y;              // warp id within a block
    const int lid           = threadIdx.x;              // lane id within a warp
    const int tid           = wid * blockDim.x + lid;   // thread id within a block
    const int threads_block = blockDim.x * blockDim.y;  // WARP_SIZE * WARPS_BLOCK
    const int A_row         = bid;
    const int C_size        = A_nrow * C_ncol;
    const int C_ncol_elems  = C_ncol * FRAG_DIM;
    const int C_bound       = C_size * FRAG_SIZE;

    __shared__ int  glbcol_sm[FRAG_DIM];  // 16
    __shared__ half A_sm[FRAG_SIZE];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM, FRAG_DIM, OutType> acc_frag;

    const auto A_row_begin = ld(&A_rowptr[A_row]);
    const auto A_row_end   = ld(&A_rowptr[min(A_row + 1, A_nrow)]);
    const auto dummy       = C_ncol * FRAG_DIM + 1;
    const auto base_idx    = lid / 4 + lid % 4 * 32;
    BMP256     bmp256;

    const int col_start      = wid;
    const int col_stride     = WARPS_BLOCK;
    const int C_ncol_rounded = round_up(C_ncol, col_stride);
    for (int curr_col = col_start; curr_col < C_ncol_rounded; curr_col += col_stride) {
        nvcuda::wmma::fill_fragment(a_frag, 0.0);
        nvcuda::wmma::fill_fragment(b_frag, 0.0);
        nvcuda::wmma::fill_fragment(acc_frag, 0.0);
        const auto out_idx = A_row * C_ncol_elems * FRAG_DIM + curr_col * FRAG_DIM;

        for (int tile_idx = A_row_begin; tile_idx < A_row_end; tile_idx++) {
            load_bmp256(bmp256, A_bitmap, tile_idx);

#pragma unroll
            for (int idx = tid; idx < FRAG_DIM; idx += threads_block) {
                glbcol_sm[idx] = dummy;
            }
#pragma unroll
            for (int idx = tid; idx < FRAG_SIZE; idx += FRAG_SIZE) {
                A_sm[idx] = 0.0;
            }
            __syncthreads();

            // fill the sparse tile and its global column indices
            decode_bmp256(tid, threads_block, bmp256, tile_idx, A_offset, A_values, A_glbcol, A_sm, glbcol_sm);

            __syncthreads();

            if (curr_col < C_ncol) {
                // load A from shared mem
                nvcuda::wmma::load_matrix_sync(a_frag, A_sm, FRAG_DIM);
                // write from global mem to registers
                load_dense_tile256(b_frag, base_idx, glbcol_sm, B_values, curr_col, C_ncol_elems, C_bound);
            }
            __syncthreads();

            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        if (curr_col < C_ncol) {
            nvcuda::wmma::store_matrix_sync(C_values + out_idx, acc_frag, C_ncol_elems, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int2 get_out_idx(int A_row1, int A_row2, int C_ncol_elems, int wid, int lid)
{
    const int lid_x_2     = lid << 1;
    const int row_in_tile = lid_x_2 >> 3;                               // Equivalent to / 8 (TILE_DIM)
    const int col_in_tile = lid_x_2 & 7;                                // Equivalent to % 8 (TILE_DIM)
    const int tile_offset = (row_in_tile * C_ncol_elems) + (wid << 3);  // wid * 8 (TILE_DIM)

    const int base_idx = (A_row1 * C_ncol_elems << 3) + tile_offset;  // * 8 (TILE_DIM)
    const int y_idx1   = base_idx + col_in_tile;
    // Add C_ncol_elems * 8 for next row
    const int y_idx2 = base_idx + (C_ncol_elems << 3) + col_in_tile;

    return make_int2(y_idx1, y_idx2);
}

template<typename IndexType, typename InputValueType>
__device__ __forceinline__ void decode_bmp64(const int             tid,
                                             const int             threads_block,
                                             const bmp64_t*        A_bmps,
                                             const int*            tile_idx,
                                             const IndexType*      A_offset,
                                             const InputValueType* A_values,
                                             const IndexType*      A_glbcol,
                                             half*                 A_sm,
                                             int*                  glbcol_sm)
{
    for (int idx = tid; idx < TILE_SIZE * 2; idx += threads_block) {
        const int     seg_idx  = idx / TILE_SIZE;
        const int     bit_idx  = idx % TILE_SIZE;
        const bmp64_t bmp64    = A_bmps[seg_idx];
        const bmp64_t bit_mask = bmp64_t(1) << bit_idx;
        const bool    valid    = bmp64 & bit_mask;

        if (valid) {
            const int bitcnt     = cntbit(bmp64 << (64 - bit_idx));
            const int A_idx      = ld(&A_offset[tile_idx[seg_idx]]) + bitcnt;
            const int local_col  = seg_idx * TILE_DIM + bit_idx % TILE_DIM;
            A_sm[idx]            = as_half(ld(&A_values[A_idx]));
            glbcol_sm[local_col] = ld(&A_glbcol[A_idx]);
        }
    }
}
template<typename InputValueType>
__device__ __forceinline__ void load_sparse_tile64(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major>& a_frag,
    const InputValueType*                                                                                        A_sm,
    const int lid_x_2)
{
#pragma unroll
    for (int m = 0; m < 2; m++) {
        int src_idx = lid_x_2 + m * TILE_SIZE;
        load_frag(a_frag, A_sm, m, src_idx);
    }
}
template<typename InputValueType>
__device__ __forceinline__ void load_dense_tile64(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major>& b_frag,
    const int*            glbcol_sm,
    const InputValueType* B_values,
    const bmp64_t*        B_elem_idx,
    const int             wid,
    const int             C_ncol_elems,
    const int             C_bound)
{
#pragma unroll
    for (int m = 0; m < 2; m++) {
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const int  dst_idx       = B_elem_idx[n];
            const auto offset        = m == 0 ? 0 : TILE_DIM;
            const auto local_idx     = dst_idx / TILE_DIM + offset;
            const int  dense_row_idx = glbcol_sm[local_idx];
            const int  src_idx       = dense_row_idx * C_ncol_elems + wid * TILE_DIM + dst_idx % TILE_DIM;
            b_frag.x[6 * m + n]      = src_idx < C_bound ? as_half(ld(&B_values[src_idx])) : zero<half>();
        }
    }
}

template<typename IndexType, typename BitmapType, typename InputValueType, typename OutputValueType>
    requires SparseOpType<IndexType, BitmapType, InputValueType, OutputValueType>
__global__ void spmm_shr64(int A_nrow,
                           int A_nnz,
                           int C_ncol,
                           const IndexType* __restrict__ A_rowptr,
                           const IndexType* __restrict__ A_colidx,
                           const IndexType* __restrict__ A_glbcol,
                           const IndexType* __restrict__ A_offset,
                           const BitmapType* __restrict__ A_bitmap,
                           const InputValueType* __restrict__ A_values,
                           const InputValueType* __restrict__ B_values,
                           OutputValueType* C_values)
{
    using OutType = std::conditional_t<std::is_same_v<OutputValueType, half>, half, float>;

    const int bid           = blockIdx.x;              // block id == row id / 2
    const int wid           = threadIdx.y;             // warp id within a block
    const int lid           = threadIdx.x;             // lane id within a warp
    const int tid           = wid * blockDim.x + lid;  // thread id within a block
    const int threads_block = blockDim.x * blockDim.y;

    const auto A_row1       = bid * 2;
    const auto A_row2       = bid * 2 + 1;
    const auto C_size       = A_nrow * C_ncol;
    const auto C_bound      = C_size * TILE_SIZE;
    const auto C_ncol_elems = C_ncol * TILE_DIM;

    const bmp64_t lid_x_2       = bmp64_t(lid) << 1;  // lid times 2
    const bmp64_t B_trans_4x8   = ((lid & 3) << 4) + (lid >> 2);
    const bmp64_t B_elem_idx[2] = {B_trans_4x8, B_trans_4x8 + 8};

    __shared__ int  glbcol_sm[FRAG_DIM];  // 8 * 2
    __shared__ half A_sm[TILE_SIZE * 2];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM, FRAG_DIM, OutType> acc_frag;

    const auto A_row_begin1 = A_row1 < A_nrow ? A_rowptr[A_row1] : A_rowptr[A_nrow];
    const auto A_row_begin2 = A_row2 < A_nrow ? A_rowptr[A_row2] : A_rowptr[A_nrow];
    const auto A_row_end1   = A_row1 < A_nrow ? A_rowptr[A_row1 + 1] : A_rowptr[A_nrow];
    const auto A_row_end2   = A_row2 < A_nrow ? A_rowptr[A_row2 + 1] : A_rowptr[A_nrow];

    const auto row_len1    = A_row_end1 - A_row_begin1;
    const auto row_len2    = A_row_end2 - A_row_begin2;
    const auto max_row_len = max(row_len1, row_len2);
    const auto dummy       = C_ncol * FRAG_DIM + 1;

    bmp64_t A_bmps[2];
    int     tile_idx[2];

    const int col_start      = wid;
    const int col_stride     = WARPS_BLOCK;
    const int C_ncol_rounded = round_up(C_ncol, col_stride);

    for (int curr_col = col_start; curr_col < C_ncol_rounded; curr_col += col_stride) {
        nvcuda::wmma::fill_fragment(a_frag, 0.0);
        nvcuda::wmma::fill_fragment(b_frag, 0.0);
        nvcuda::wmma::fill_fragment(acc_frag, 0.0);

        for (int i = 0; i < max_row_len; i++) {
            tile_idx[0] = A_row_begin1 + i;
            tile_idx[1] = A_row_begin2 + i;
            A_bmps[0]   = i < row_len1 ? ld(&A_bitmap[tile_idx[0]]) : 0;
            A_bmps[1]   = i < row_len2 ? ld(&A_bitmap[tile_idx[1]]) : 0;

#pragma unroll
            for (int idx = tid; idx < FRAG_DIM; idx += threads_block) {
                glbcol_sm[idx] = dummy;
            }
#pragma unroll
            for (int idx = tid; idx < TILE_SIZE * 2; idx += FRAG_SIZE) {
                A_sm[idx] = zero<half>();
            }
            __syncthreads();
            // prepare A tile and its global column indices in shared memory
            decode_bmp64(tid, threads_block, A_bmps, tile_idx, A_offset, A_values, A_glbcol, A_sm, glbcol_sm);

            __syncthreads();

            if (curr_col < C_ncol) {
                // load A tile from shared memory
                load_sparse_tile64(a_frag, A_sm, lid_x_2);

                // load B tile based on global column indices in shared memory
                load_dense_tile64(b_frag, glbcol_sm, B_values, B_elem_idx, curr_col, C_ncol_elems, C_bound);
            }
            __syncthreads();
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        const int2 y_idx = get_out_idx(A_row1, A_row2, C_ncol_elems, curr_col, lid);

        if (curr_col < C_ncol) {
            if (A_row1 < A_nrow) {
                store_frag(C_values, acc_frag, 0, y_idx.x);
            }
            if (A_row2 < A_nrow) {
                store_frag(C_values, acc_frag, 6, y_idx.y);
            }
        }
    }
}

}  // namespace thunder

#pragma once
#include <cuda.h>
#include <mma.h>

namespace thunder {

template<typename IndexType, typename BitmapType, typename InputValueType, typename OutputValueType>
    requires SparseOpType<IndexType, BitmapType, InputValueType, OutputValueType>
__global__ void sddmm_shr256(int A_nrow,
                             int A_nnz,
                             int C_ncol,
                             const IndexType* __restrict__ A_rowptr,
                             const IndexType* __restrict__ A_colidx,
                             const IndexType* __restrict__ A_glbcol,
                             const IndexType* __restrict__ A_offset,
                             const BitmapType* __restrict__ A_bitmap,
                             const InputValueType* __restrict__ B_values,
                             const InputValueType* __restrict__ C_values,
                             OutputValueType* __restrict__ A_values)
{
    using OutType = std::conditional_t<std::is_same_v<OutputValueType, half>, half, float>;

    const auto bid           = blockIdx.x;              // block id == row id
    const auto wid           = threadIdx.y;             // warp id within a block
    const auto lid           = threadIdx.x;             // lane id within a warp
    const auto tid           = wid * blockDim.x + lid;  // thread id within a block
    const auto threads_block = blockDim.x * blockDim.y;

    const int A_row        = bid;
    const int C_size       = A_nrow * C_ncol;
    const int C_ncol_elems = C_ncol * FRAG_DIM;
    const int C_bound      = C_size * FRAG_SIZE;
    const int A_rowidx     = A_row * FRAG_DIM;

    __shared__ int glbcol_sm[FRAG_DIM];  // 16
    __shared__ int A_idx_sm[FRAG_SIZE];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> x_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::col_major> y_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM, FRAG_DIM, OutType> acc_frag;

    nvcuda::wmma::fill_fragment(x_frag, 0.0);
    nvcuda::wmma::fill_fragment(y_frag, 0.0);
    nvcuda::wmma::fill_fragment(acc_frag, 0.0);

    const auto A_row_begin = ld(&A_rowptr[A_row]);
    const auto A_row_end   = ld(&A_rowptr[min(A_row + 1, A_nrow)]);

    int A_elem_idx[2];
    int B_elem_idx[2];

    A_elem_idx[0] = lid / 4 * 16 + lid % 4 * 2;
    A_elem_idx[1] = A_elem_idx[0] + 1;
    B_elem_idx[0] = lid / 4 + lid % 4 * 32;
    B_elem_idx[1] = B_elem_idx[0] + 16;

    BMP256 bmp256;

    for (auto tile_idx = A_row_begin; tile_idx < A_row_end; tile_idx++) {
        load_bmp256(bmp256, A_bitmap, tile_idx);

        for (int idx = tid; idx < FRAG_DIM; idx += threads_block) {
            glbcol_sm[lid] = A_nrow * FRAG_DIM + 1;
        }
#pragma unroll
        for (int idx = tid; idx < FRAG_SIZE; idx += threads_block) {
            A_idx_sm[idx] = A_nnz + 1;
        }
        __syncthreads();

#pragma unroll
        for (int idx = tid; idx < FRAG_SIZE; idx += threads_block) {
            const int     seg_idx  = idx / TILE_SIZE;
            const int     bit_idx  = idx % TILE_SIZE;
            const bmp64_t bmp64    = bmp256[seg_idx];
            const bmp64_t bit_mask = bmp64_t(1) << bit_idx;
            const bool    valid    = bmp64 & bit_mask;

            if (valid) {
                const int bitcnt = cntbit(bmp64 << (64 - bit_idx));
                const int A_idx  = ld(&A_offset[tile_idx * NUM_BMP64 + seg_idx]) + bitcnt;

                A_idx_sm[idx]        = A_idx;
                const int local_col  = idx % FRAG_DIM;
                glbcol_sm[local_col] = ld(&A_glbcol[A_idx]);
            }
        }
        __syncthreads();

        for (int warp_iter = 0; warp_iter < C_ncol; warp_iter++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int A_dst_idx    = A_frag_offsets[i] + A_elem_idx[j];
                    const int dense_rowidx = A_dst_idx / FRAG_DIM;
                    const int dense_colidx = A_dst_idx % FRAG_DIM;
                    const int source_idx =
                        (A_rowidx + dense_rowidx) * C_ncol_elems + warp_iter * FRAG_DIM + dense_colidx;
                    x_frag.x[2 * i + j] = source_idx < C_bound ? as_half(ld(&B_values[source_idx])) : zero<half>();
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int B_dst_idx    = B_frag_offsets[i] + A_elem_idx[j];
                    const int dense_rowidx = glbcol_sm[B_dst_idx / FRAG_DIM];
                    const int dense_colidx = B_dst_idx % FRAG_DIM;
                    const int source_idx   = dense_rowidx * C_ncol_elems + warp_iter * FRAG_DIM + dense_colidx;
                    y_frag.x[2 * i + j]    = source_idx < C_bound ? as_half(ld(&C_values[source_idx])) : zero<half>();
                }
            }
            __syncthreads();

            nvcuda::wmma::mma_sync(acc_frag, x_frag, y_frag, acc_frag);
        }  // <--- ending of warp iteration.

        // Output the results to sparse matrix.
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int A_dst_idx = A_frag_offsets[i] + A_elem_idx[j];
                if (A_idx_sm[A_dst_idx] < A_nnz) {
                    const int index = A_idx_sm[A_dst_idx];
                    A_values[index] = acc_frag.x[2 * i + j];
                }
            }
        }  //<-- ending of storing output to global memory.
        nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    }
}

template<typename IndexType, typename BitmapType, typename InputValueType, typename OutputValueType>
    requires SparseOpType<IndexType, BitmapType, InputValueType, OutputValueType>
__global__ void sddmm_shr64(int A_nrow,
                            int A_nnz,
                            int C_ncol,
                            const IndexType* __restrict__ A_rowptr,
                            const IndexType* __restrict__ A_colidx,
                            const IndexType* __restrict__ A_glbcol,
                            const IndexType* __restrict__ A_offset,
                            const BitmapType* __restrict__ A_bitmap,
                            const InputValueType* __restrict__ B_values,
                            const InputValueType* __restrict__ C_values,
                            OutputValueType* __restrict__ A_values)
{
    const auto bid           = blockIdx.x;              // block id == row id
    const auto wid           = threadIdx.y;             // warp id within a block
    const auto lid           = threadIdx.x;             // lane id within a warp
    const auto tid           = wid * blockDim.x + lid;  // thread id within a block
    const auto threads_block = blockDim.x * blockDim.y;

    const int A_row1       = bid * 2;
    const int A_row2       = bid * 2 + 1;
    const int C_size       = A_nrow * C_ncol;
    const int C_bound      = C_size * TILE_SIZE;
    const int C_ncol_elems = C_ncol * TILE_DIM;

    const bmp64_t lid_times_2   = bmp64_t(lid) << 1;
    const bmp64_t bitmask1      = bmp64_t(1) << lid_times_2;
    const bmp64_t A_bitmasks[2] = {bitmask1, bitmask1 << 1};
    const bmp64_t A_reverse[2]  = {64 - lid_times_2, 63 - lid_times_2};
    const bmp64_t A_elem_idx[2] = {lid_times_2, lid_times_2 + 1};
    const int     A_rowidx[2]   = {A_row1 * TILE_DIM, A_row2 * TILE_DIM};

    __shared__ int glbcol_sm[FRAG_DIM];  // 16
    __shared__ int A_idx_sm[TILE_SIZE * 2];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::row_major> x_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM, half, nvcuda::wmma::col_major> y_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM, FRAG_DIM, float> acc_frag;

    nvcuda::wmma::fill_fragment(x_frag, 0.0);
    nvcuda::wmma::fill_fragment(y_frag, 0.0);
    nvcuda::wmma::fill_fragment(acc_frag, 0.0);

    const auto A_row_begin1 = A_row1 < A_nrow ? A_rowptr[A_row1] : A_rowptr[A_nrow];
    const auto A_row_begin2 = A_row2 < A_nrow ? A_rowptr[A_row2] : A_rowptr[A_nrow];
    const auto row_len1     = (A_row1 < A_nrow) ? A_rowptr[A_row1 + 1] - A_row_begin1 : 0;
    const auto row_len2     = (A_row2 < A_nrow) ? A_rowptr[A_row2 + 1] - A_row_begin2 : 0;
    const auto max_row_len  = max(row_len1, row_len2);
    const auto dummy        = C_ncol * FRAG_DIM + 1;

    bmp64_t A_bmps[2];
    int     tile_idx[2];

    auto load_tile = [&](int tile_idx, bmp64_t A_bmp, int base) {
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const bool valid = A_bmp & A_bitmasks[n];
            if (valid) {
                const int elem_idx = A_elem_idx[n];
                const int num_bits = cntbit(A_bmp << A_reverse[n]);
                const int A_idx    = ld(&A_offset[tile_idx]) + num_bits;

                A_idx_sm[base + elem_idx] = A_idx;
                const int offset          = base == 0 ? 0 : TILE_DIM;
                const int local_col       = elem_idx % TILE_DIM + offset;
                glbcol_sm[local_col]      = ld(&A_glbcol[A_idx]);
            }
        }
    };

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
        for (int idx = tid; idx < TILE_SIZE * 2; idx += threads_block) {
            A_idx_sm[idx] = A_nnz + 1;
        }
        __syncthreads();
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
                A_idx_sm[idx]        = A_idx;
                glbcol_sm[local_col] = ld(&A_glbcol[A_idx]);
            }
        }
        __syncthreads();

        for (int warp_iter = 0; warp_iter < C_ncol; warp_iter++) {
// fill x fragment
#pragma unroll
            for (int m = 0; m < 2; m++) {
#pragma unroll
                for (int n = 0; n < 2; n++) {
                    const auto dst_idx      = A_elem_idx[n];
                    const auto dense_rowidx = dst_idx / TILE_DIM;
                    const auto dense_colidx = dst_idx % TILE_DIM;
                    const auto src_idx =
                        (A_rowidx[m] + dense_rowidx) * C_ncol_elems + warp_iter * TILE_DIM + dense_colidx;
                    x_frag.x[6 * m + n] = src_idx < C_bound ? as_half(ld(&B_values[src_idx])) : zero<half>();
                }
            }
            __syncthreads();
// fill y fragment
#pragma unroll
            for (int m = 0; m < 2; m++) {
#pragma unroll
                for (int n = 0; n < 2; n++) {
                    const auto dst_idx      = A_elem_idx[n];
                    const auto local_idx    = dst_idx / TILE_DIM + m * TILE_DIM;
                    const auto dense_rowidx = glbcol_sm[local_idx];
                    const auto dense_colidx = dst_idx % TILE_DIM;
                    const auto src_idx      = dense_rowidx * C_ncol_elems + warp_iter * TILE_DIM + dense_colidx;
                    y_frag.x[6 * m + n]     = src_idx < C_bound ? as_half(ld(&B_values[src_idx])) : zero<half>();
                }
            }
            __syncthreads();
            nvcuda::wmma::mma_sync(acc_frag, x_frag, y_frag, acc_frag);
        }  // <--- ending of warp iteration.
           // output the results to sparse matrix.
#pragma unroll
        for (int m = 0; m < 2; m++) {
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const int A_dst_idx = A_elem_idx[n] + m * TILE_SIZE;
                if (A_idx_sm[A_dst_idx] < A_nnz) {
                    const int index = A_idx_sm[A_dst_idx];
                    A_values[index] = acc_frag.x[6 * m + n];
                }
            }
        }  //<-- ending of storing output to global memory.
        nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    }
}

}  // namespace thunder
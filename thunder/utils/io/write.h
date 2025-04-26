#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace thunder {

template<typename CooMatrix>
void write_coo2binary(const CooMatrix& A_coo, char* filename)
{
    std::ofstream output_file(std::string(filename), std::ios::binary);
    using IndexType = CooMatrix::index_type;

    thrust::host_vector<IndexType> rowIdx = A_coo.row_indices;
    thrust::host_vector<IndexType> colIdx = A_coo.column_indices;
    std::vector<IndexType>         edge_list(A_coo.num_entries * 2);
#pragma omp parallel for
    for (int i = 0; i < A_coo.num_entries; i++) {
        edge_list[2 * i]     = rowIdx[i];
        edge_list[2 * i + 1] = colIdx[i];
    }

    if (!output_file.is_open()) {
        printf("cannot open the output binary file!\n");
        std::exit(1);
    }
    output_file.write(reinterpret_cast<char*>(edge_list.data()), sizeof(IndexType) * 2 * A_coo.num_entries);
    output_file.close();
}

template<typename CsrMatrix>
bool write_into_csr(const CsrMatrix& mat, std::string output)
{
    using IndexType                        = CsrMatrix::index_type;
    auto                           nrow    = mat.num_rows;
    auto                           nnz     = mat.num_entries;
    thrust::host_vector<IndexType> row_ptr = mat.row_pointers;
    thrust::host_vector<IndexType> col_idx = mat.column_indices;
    FILE*                          fp      = fopen(output.c_str(), "wb");
    if (fp == NULL) {
        fputs("file error", stderr);
        return false;
    }
    std::cout << "writing to " << output << std::endl;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    fwrite(&nrow, sizeof(IndexType), 1, fp);
    fwrite(&nnz, sizeof(IndexType), 1, fp);
    fwrite(row_ptr.data(), sizeof(IndexType), nrow + 1, fp);
    fwrite(col_idx.data(), sizeof(IndexType), nnz, fp);

    fclose(fp);

    return true;
};

template<typename CsrMatrix>
bool write_into_edgelist(const CsrMatrix& mat, std::string out)
{
    std::ofstream output(out);
    if (!output.is_open()) {
        std::cout << "cannot open the output file!" << std::endl;
        return false;
    }
    auto                     nrow    = mat.num_rows;
    auto                     nnz     = mat.num_entries;
    thrust::host_vector<int> row_ptr = mat.row_pointers;
    thrust::host_vector<int> col_idx = mat.column_indices;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    for (unsigned i = 0; i < nrow; i++) {
        for (unsigned j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            output << i << " " << col_idx[j] << '\n';
        }
    }
    output.close();

    return true;
}

template<typename CsrMatrix>
bool write_into_bel(const CsrMatrix& mat, std::string out)
{
    std::ofstream output(out);
    if (!output.is_open()) {
        std::cout << "cannot open the output file!" << std::endl;
        return false;
    }

    using IndexType                        = CsrMatrix::index_type;
    auto                           nrow    = mat.num_rows;
    auto                           nnz     = mat.num_entries;
    thrust::host_vector<IndexType> row_ptr = mat.row_pointers;
    thrust::host_vector<IndexType> col_idx = mat.column_indices;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    std::vector<IndexType> edge_list(nnz * 2);
#pragma omp parallel for
    for (unsigned i = 0; i < nrow; i++) {
        for (unsigned j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            edge_list[2 * j]     = i;
            edge_list[2 * j + 1] = col_idx[j];
        }
    }
    output.write(reinterpret_cast<char*>(edge_list.data()), sizeof(IndexType) * 2 * nnz);
    output.close();

    return true;
}

template<typename CsrMatrix>
bool write_into_mtx(const CsrMatrix& mat, std::string out)
{
    std::ofstream output(out);
    if (!output.is_open()) {
        std::cout << "cannot open the output file!" << std::endl;
        return false;
    }
    auto                     nrow    = mat.num_rows;
    auto                     nnz     = mat.num_entries;
    thrust::host_vector<int> row_ptr = mat.row_pointers;
    thrust::host_vector<int> col_idx = mat.column_indices;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    output << "%%MatrixMarket matrix coordinate pattern general\n";
    output << nrow << " " << nrow << " " << nnz << '\n';
    for (unsigned i = 0; i < nrow; i++) {
        for (unsigned j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            output << i + 1 << " " << col_idx[j] + 1 << '\n';
        }
    }
    output.close();

    return true;
}

template<typename SHR>
bool write_into_shr(const SHR& mat, std::string output)
{
    auto nrow  = mat.num_rows;
    auto ncol  = mat.num_cols;
    auto ntile = mat.num_tiles;
    auto nnz   = mat.num_entries;

    static constexpr int nBmp64 = SHR::bmp64_count;

    thrust::host_vector<int>     rowptr       = mat.row_pointers;
    thrust::host_vector<int>     colidx       = mat.column_indices;
    thrust::host_vector<bmp64_t> bitmaps      = mat.bitmaps;
    thrust::host_vector<int>     tile_offsets = mat.tile_offsets;
    thrust::host_vector<int>     glb_colidx   = mat.column_indices_global;

    FILE* fp = fopen(output.c_str(), "wb");
    if (fp == NULL) {
        fputs("file error", stderr);
        return false;
    }
    std::cout << "writing to " << output << std::endl;
    ASSERT(rowptr[nrow] == ntile && "rowptr[nrow] != nnz");

    fwrite(&nrow, sizeof(int), 1, fp);
    fwrite(&ncol, sizeof(int), 1, fp);
    fwrite(&ntile, sizeof(int), 1, fp);
    fwrite(&nnz, sizeof(int), 1, fp);
    fwrite(&nBmp64, sizeof(int), 1, fp);

    fwrite(rowptr.data(), sizeof(int), nrow + 1, fp);
    fwrite(colidx.data(), sizeof(int), ntile, fp);
    fwrite(bitmaps.data(), sizeof(bmp64_t), ntile * nBmp64, fp);
    fwrite(tile_offsets.data(), sizeof(int), ntile * nBmp64 + 1, fp);
    fwrite(glb_colidx.data(), sizeof(int), nnz, fp);

    fclose(fp);

    return true;
}

template<typename CsrMatrix>
void write_matrix_file(CsrMatrix& d_csr_A, std::string output)
{
    if (output.empty()) {
        return;  // nothing happens
    }
    if (string_end_with(output, ".el")) {
        std::cout << "converting to text edge list" << std::endl;
        write_into_edgelist(d_csr_A, output);
    }
    else if (string_end_with(output, ".csr")) {
        std::cout << "converting to CSR format" << std::endl;
        write_into_csr(d_csr_A, output);
    }
    else if (string_end_with(output, ".mtx")) {
        std::cout << "converting to MTX format" << std::endl;
        write_into_mtx(d_csr_A, output);
    }
    else if (string_end_with(output, ".bel")) {
        std::cout << "converting to binary edge list" << std::endl;
        write_into_bel(d_csr_A, output);
    }
    else {
        printf("file format is not supported\n");
        std::exit(1);
    }
}

}  // namespace thunder
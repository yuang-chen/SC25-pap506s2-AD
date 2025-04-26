#pragma once

#include <thrust/shuffle.h>

namespace thunder {

template<int seed, typename T>
struct RandomGenerator {
    T min_val, max_val;

    explicit RandomGenerator(T _min, T _max): min_val(_min), max_val(_max) {}

    __host__ __device__ T operator()(int n) const
    {
        thrust::default_random_engine rng(seed);
        if constexpr (std::is_integral<T>::value) {
            thrust::uniform_int_distribution<T> dist(min_val, max_val);
            rng.discard(n);  // Ensure different seeds for different indices
            return dist(rng);
        }
        else if constexpr (std::is_floating_point<T>::value) {
            thrust::uniform_real_distribution<T> dist(min_val, max_val);
            rng.discard(n);  // Ensure different seeds for different indices
            return dist(rng);
        }
        else if constexpr (std::is_same_v<T, half>) {
            thrust::uniform_real_distribution<float> dist(static_cast<float>(min_val), static_cast<float>(max_val));
            rng.discard(n);  // Ensure different seeds for different indices
            return static_cast<half>(dist(rng));
        }
        else {
            static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "Unsupported type");
        }
    }
};

template<int seed = 2049, typename Vector, typename T = Vector::value_type>
void randomize(Vector& vec, const thrust::pair<T, T>& range)
{
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(vec.size()),
                      vec.begin(),
                      RandomGenerator<seed, T>(thrust::get<0>(range), thrust::get<1>(range)));
}

template<typename CSR, typename IndexType = CSR::index_type, typename ValueType = CSR::value_type>
void generate_random_csr(CSR&                                      mat,
                         IndexType                                 num_rows,
                         IndexType                                 num_cols,
                         IndexType                                 num_entries,
                         const thrust::pair<ValueType, ValueType>& range = {1.1, 2.2})
{

    // row_pointers.resize(num_rows + 1, 0);
    // column_indices.resize(nnz);
    // values.resize(nnz);
    // mat.resize(num_rows, num_cols, num_entries);

    thrust::device_vector<IndexType> row_pointers(num_rows + 1, 0);

    const IndexType max_possible = num_rows * num_cols;
    num_entries                  = thrust::min(num_entries, max_possible);

    // 1. Generate all possible unique indices
    thrust::device_vector<IndexType> linear_indices(max_possible);
    thrust::sequence(linear_indices.begin(), linear_indices.end());

    // 2. Shuffle and select first num_entries
    thrust::default_random_engine rng(2049);
    thrust::shuffle(linear_indices.begin(), linear_indices.end(), rng);
    linear_indices.resize(num_entries);

    // 3. Convert linear indices to (row, col) pairs
    thrust::device_vector<IndexType> row_indices(num_entries);
    thrust::device_vector<IndexType> column_indices(num_entries);

    auto convert = [num_cols] __device__(IndexType idx) { return thrust::make_tuple(idx / num_cols, idx % num_cols); };

    thrust::transform(linear_indices.begin(),
                      linear_indices.end(),
                      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
                      convert);

    // 4. Sort by row then column
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin()));

    thrust::sort(zip_begin, zip_begin + num_entries);

    thrust::device_vector<ValueType> values(num_entries);
    randomize<7799>(values, range);

    get_row_pointers_from_indices(row_pointers, row_indices);

    mat.num_rows    = num_rows;
    mat.num_cols    = num_cols;
    mat.num_entries = num_entries;

    mat.row_pointers   = std::move(row_pointers);
    mat.column_indices = std::move(column_indices);
    mat.values         = std::move(values);
}

template<typename CSR, typename IndexType = CSR::index_type, typename ValueType = CSR::value_type>
void generate_random_matrix(CSR&                                      mat,
                            IndexType                                 num_rows,
                            IndexType                                 num_cols,
                            float                                     sparsity,
                            const thrust::pair<ValueType, ValueType>& range = thrust::make_pair(0.0f, 1.0f))
{
    if (sparsity <= 0.0f || sparsity >= 1.0f) {
        return;
    }
    printf("generate random matrix with sparsity: %f\n", sparsity);

    // Calculate number of non-zero elements based on sparsity
    const IndexType total_elements = num_rows * num_cols;
    const IndexType num_entries    = static_cast<IndexType>(total_elements * (1.0f - sparsity));

    // Call the base random matrix generator
    generate_random_csr(mat, num_rows, num_cols, num_entries, range);
}

}  // namespace thunder
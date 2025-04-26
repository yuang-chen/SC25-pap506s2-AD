#pragma once
#include <experimental/type_traits>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <bitset>
#include <limits>

namespace thunder {


template<typename T>
struct MismatchInfo {
    size_t index;
    T      value1;
    T      value2;
};

template<typename Vector, typename T = Vector::value_type>
void report_mismatches(const Vector& vec1,
                       const Vector& vec2,
                       float         tolerance = 1e-3,
                       size_t        limit     = std::numeric_limits<size_t>::max())
{
    std::vector<T> array1(vec1.size());
    std::vector<T> array2(vec2.size());
    if (array1.size() > array2.size()) {
        std::cerr << "Array1 should not have larger size" << '\n';
    }
    limit = std::min(limit, array1.size());
    thrust::copy(vec1.begin(), vec1.begin() + limit, array1.begin());
    thrust::copy(vec2.begin(), vec2.begin() + limit, array2.begin());
    std::vector<MismatchInfo<float>> mismatches;
    for (size_t i = 0; i < array1.size(); i++) {
        float diff, val1, val2;
        if constexpr (std::is_same_v<T, half>) {
            val1 = __half2float(array1[i]);
            val2 = __half2float(array2[i]);
            diff = std::fabs(val1 - val2) / val1;
        }
        else {
            diff = std::fabs(array1[i] - array2[i]) / array1[i];
        }

        if (diff > tolerance) {
            mismatches.push_back({i, val1, val2});
        }
    }

    if (mismatches.size() == 0) {
        std::cout << "Results with tolerance of " << tolerance << " are verified. No mismatches found" << '\n';
        return;
    }

    // Report mismatches
    limit = std::min(limit, mismatches.size());
    for (int i = 0; i < limit; i++) {
        std::cout << "Mismatch at index " << mismatches[i].index << ": " << mismatches[i].value1 << " vs "
                  << mismatches[i].value2 << '\n';
    }

    if (limit < mismatches.size()) {
        std::cout << "...\n";
        std::cout << "Mismatch at index " << mismatches.back().index << ": " << mismatches.back().value1 << " vs "
                  << mismatches.back().value2 << '\n';
    }
    std::cout << "total mismatches: " << mismatches.size() << '\n';
}


template<BMP_SIZE BmpSize>
constexpr auto describe_ops_types()
{
    std::string bmp_type;

    if constexpr (BmpSize == BMP_SIZE::BMP64) {
        bmp_type = "[DIA]";
    }
    else if constexpr (BmpSize == BMP_SIZE::BMP256) {
        bmp_type = "[FULL]";
    }
    else {
        return "[unknown]";
    }

    return bmp_type;
}

}  // namespace thunder
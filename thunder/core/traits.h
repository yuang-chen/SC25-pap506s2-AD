#pragma once

#include <cuda_fp16.h>
#include <type_traits>

namespace thunder {

template<typename T>
constexpr bool is_half_or_float = std::is_same_v<T, half> || std::is_same_v<T, float>;

template<typename IndexType, typename BitmapType, typename InputValueType, typename OutputValueType>
concept SparseOpType = std::is_same_v<IndexType, int> && std::is_same_v<BitmapType, bmp64_t>
                       && is_half_or_float<InputValueType> && is_half_or_float<OutputValueType>;

}  // namespace thunder
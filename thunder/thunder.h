#pragma once
// clang-format off
// Core includes
#include "core/types.h"
#include "core/traits.h"
#include "core/macros.h"
#include "core/memory.h"
#include "core/constants.h"


// Compute operations
#include "core/compute/bitwise.h"
#include "core/compute/fragments.h"
#include "core/compute/intrinsics.h"
#include "core/compute/arithmetic.h"

// Matrix formats
#include "formats/bitcoo.h"
#include "formats/bitcsr.h"
#include "formats/bitshr.h"
#include "formats/coo.h"
#include "formats/csr.h"
#include "formats/dense.h"
#include "formats/shr.h"


// Utilities - Functors
#include "utils/functors/bitmap_ops.h"
#include "utils/functors/matrix_ops.h"
#include "utils/functors/predicate_ops.h"

// Utilities - Others
#include "utils/timer.h"
#include "utils/csr_helpers.h"
#include "utils/random.h"
#include "utils/option.h"
#include "utils/precision.h"
#include "utils/report.h"

// Utilities - IO
#include "utils/io/mmio.h"
#include "utils/io/read.h"
#include "utils/io/write.h"


// Transform Matrix
#include "transforms/convert.h"
#include "transforms/reorder.h"
#include "transforms/shrink.h"


// Operations
#include "operations/sddmm.h"
#include "operations/spmm.h"


// cuSPARSE
#include "cusparse/sddmm.h"
#include "cusparse/spmm.h"

namespace thunder {
// Version information
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;


}  // namespace thunder
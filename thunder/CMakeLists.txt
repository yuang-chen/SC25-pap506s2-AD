add_library(bmplib INTERFACE)

# target_include_directories(bmplib INTERFACE .)
# Use this instead
target_include_directories(bmplib
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(bmplib INTERFACE
  OpenMP::OpenMP_CXX
  CUDA::cusparse
  CUDA::cublas
#  cusparse_lt
)

if(KGRAPH)
  target_link_libraries(bmplib INTERFACE
    kgraph
    Boost::timer
    Boost::chrono
    Boost::system
    Boost::program_options)
endif()

target_compile_options(bmplib INTERFACE
  $<$<CONFIG:Debug>:-G -g>
  $<$<COMPILE_LANGUAGE:CUDA>:
  -Xcompiler=-fopenmp # OpenMP flags for CUDA compiler
  --extended-lambda
  --expt-relaxed-constexpr
  -w # Suppress warning -- though warning is usually helpful
  >
)

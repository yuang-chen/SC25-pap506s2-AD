
add_executable(spmm spmm.cu)
add_executable(sddmm sddmm.cu)


target_link_libraries(spmm PRIVATE bmplib)
target_link_libraries(sddmm PRIVATE bmplib)


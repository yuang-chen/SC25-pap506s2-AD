# ThunderGNN on tensor cores
ThunderGNN is a framework optimized to leverage Tensor Cores (TCs) for GNN speedup. This repository contains the implementation of the two key kernels: SpMM and SDDMM.

## How to build
Cmake and CUDA are required. Modification to the CMakeLists.txt files may be necessary, e.g. to change GPU architecture.

Instructions:
1. Download the source code into a folder e.g. ThunderGNN.
3. Inside ThunderGNN and command:

```bash
mkdir build  && cd build 
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && cd ..
```
## Datasets
The accepted format for the datasets is binary CSR or text MTX. 
16 graphs are provided in the `data/gnn` folder. 

To download the SuiteSparse matrices, please firstly install the `ssgetpy` package following this [github](https://github.com/drdarshan/ssgetpy).

Then, download the datasets with the script: ` ./scripts/download_mtx.sh`


## How to run
For testing SpMM and SDDMM, the executable accepts differents arguments. 
```bash
Usage: ./build/apps/spmm ... 
              [-i input_file]
              [-o output_file]
              [-c columns_of_dense_matrix]
              [-e execution_iterations]
              [-v verify_results (1 or 0)]
              [-r reorder_algorithm (0: none, 1: row sorting)]
```

### Examples

1. Run with input file
> ./build/apps/spmm -i /input/path/data.csr

2. Convert graph from MTX to CSR
> ./build/apps/spmm -i /input/path/data.mtx -o /output/path/data.csr -e 0

3. Run with a script
> ./scripts/run_gnn.sh


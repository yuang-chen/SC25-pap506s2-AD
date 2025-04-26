#!/bin/bash


GNN=( "amazon0505" "artist" "com-amazon" "DD" "ppi" "YeastH" "amazon0601" \
    "citeseer" "cora" "OVCAR-8H" "PROTEINS_full" "soc-BlogCatalog" "Yeast" "pubmed" )

O_DIR=./logs/gnn
mkdir -p ${O_DIR}  

for DATA in ${GNN[@]} ; do 
    for COL in 16 ; do
        for APP in spmm sddmm; do
            PREFIX=${APP}.${DATA}.c${COL}
            echo ${PREFIX}
            ./build/apps/${APP} -i ./data/gnn/${DATA}.csr -e 1000 > ${O_DIR}/${PREFIX}.log 2>&1
        done
    done
done

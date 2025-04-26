#!/bin/bash

readarray -t ALL_MTX < ./scripts/namelist.txt


O_DIR=./logs/mtx
mkdir -p ${O_DIR}  


for APP in spmm sddmm; do   
    for DATA in ${ALL_MTX[@]} ; do 
        for COL in 16; do
            PREFIX=${APP}.${DATA}.c${COL}
            echo ${PREFIX}
            ./build/apps/${APP} -i ./data/mtx/${DATA}/${DATA}.mtx \
            -e 1000  -c ${COL} > ${O_DIR}/${PREFIX}.log 2>&1       
        done
    done
done


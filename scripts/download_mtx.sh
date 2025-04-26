#!/bin/bash

readarray -t ALL_MTX < ./scripts/namelist.txt

for mat in "${ALL_MTX[@]}"; do
  echo "Downloading matrix: $mat"
  ssgetpy -n "$mat" -o ./data/mtx/
done


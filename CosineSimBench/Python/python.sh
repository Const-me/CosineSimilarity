#!/bin/sh

BENCH="python3 ./cosineSimdBench.py"
export OPENBLAS_NUM_THREADS=8
for S in 128k 256k 512k 1M 2M 4M 8M 16M 32M
do
    $BENCH "$S"
done
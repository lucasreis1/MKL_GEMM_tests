#!/bin/bash

BENCHMARKS=('bf16bf16f32' 'f16f16f32' 's16s16s32' 's8u8s32' 'hgemm' 'sgemm' 'dgemm')

BENCHS_DIR="/home/lucas.reis/mkl/tests/build"

export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=8
for i in $(seq 1 20)
do
  for benchmark in ${BENCHMARKS[*]}
  do
    #$BENCHS_DIR/$benchmark 64 800 320 50
    #$BENCHS_DIR/$benchmark 64 768 512 50
    #$BENCHS_DIR/$benchmark 16 256 512 50
    $BENCHS_DIR/$benchmark 128 128 128 50
    $BENCHS_DIR/$benchmark 256 256 256 50
    $BENCHS_DIR/$benchmark 256 512 256 50
    $BENCHS_DIR/$benchmark 512 512 512 50
    $BENCHS_DIR/$benchmark 1024 1024 1024 50
    $BENCHS_DIR/$benchmark 2048 2048 2048 50
    $BENCHS_DIR/$benchmark 4096 4096 4096 50
    $BENCHS_DIR/$benchmark 8192 8192 8192 50
    $BENCHS_DIR/$benchmark 16384 16384 16384 50
    $BENCHS_DIR/$benchmark 32768 32768 32768 50
    $BENCHS_DIR/$benchmark 65536 65536 65536 50
  done
done

#!/bin/bash

BENCHMARKS=('bf16bf16f32' 'f16f16f32' 's16s16s32' 's8u8s32' 'hgemm' 'sgemm' 'dgemm')

BENCHS_DIR="/home/lucas.reis/mkl/tests/build"

export MKL_DYNAMIC=FALSE
for benchmark in ${BENCHMARKS[*]}
do
  $BENCHS_DIR/${benchmark}_store_output 128 128 128 output_${benchmark}_128x128x128_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 256 256 256 output_${benchmark}_256x256x256_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 256 256 512 output_${benchmark}_256256x512x25_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 512 512 512 output_${benchmark}_512x512x512_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 1024 1024 1024 output_${benchmark}_1024x1024x1024_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 2048 2048 2048 output_${benchmark}_2048x2048x2048_randominps.bin
  $BENCHS_DIR/${benchmark}_store_output 4096 4096 4096 output_${benchmark}_4096x4096x4096_randominps.bin
  #$BENCHS_DIR/$benchmark 8192 8192 8192 50
  #$BENCHS_DIR/$benchmark 16384 16384 16384 50
  #$BENCHS_DIR/$benchmark 32768 32768 32768 50
  #$BENCHS_DIR/$benchmark 65536 65536 65536 50
done

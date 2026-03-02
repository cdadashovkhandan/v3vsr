#!/bin/bash

set -e

# Get XLA include path from Python
XLA_DIR=$(python -c "from jax import ffi; print(ffi.include_dir())")

# Paths
SRC="src/ops_cuda.cu"
OUT="dist/_ops_cuda.so"

mkdir -p $(dirname "$OUT")

# Compile
nvcc -std=c++17 -Xcompiler -fPIC -I"$XLA_DIR" -shared "$SRC" -o "$OUT"
echo "Built $OUT"

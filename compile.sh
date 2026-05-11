#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++20 --extended-lambda -rdc=true \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  main.cu \
  kernels.cu \
  -o main

./main "${1:-0}"


# // Para Gustavito mi amigoooo 
# nvcc -O3 -std=c++20 --extended-lambda -rdc=true \
#   -arch=native \
#   main.cu kernels.cu \
#   -o main

# ./main "${1:-0}"

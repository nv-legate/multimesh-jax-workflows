#! /usr/bin/env bash

make -j src.build  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90a,code=sm_90a"
make -j pkg.debian.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90a,code=sm_90a"

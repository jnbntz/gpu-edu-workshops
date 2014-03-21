#!/bin/bash

#SBATCH -n 1
#SBATCH -n 2
#SBATCH -p gpgpu

./x.matmul_CUBLAS

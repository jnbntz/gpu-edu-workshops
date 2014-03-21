#!/bin/bash

#SBATCH -n 1
#SBATCH -t 2
#SBATCH -p gpgpu

./x.naive_transpose

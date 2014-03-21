#!/bin/bash

bin=matmul_GPU_shmem1

nvprof --output-profile $bin.timeline ./x.$bin
nvprof --analysis-metrics -o $bin.analysis ./x.$bin

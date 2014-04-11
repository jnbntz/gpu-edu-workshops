#!/bin/bash

bin=smem_transpose_opt

nvprof --output-profile $bin.timeline ./x.$bin
nvprof --analysis-metrics -o $bin.analysis ./x.$bin

#!/bin/bash

bin=smem_transpose

nvprof --output-profile $bin.timeline ./x.$bin
nvprof --analysis-metrics -o $bin.analysis ./x.$bin

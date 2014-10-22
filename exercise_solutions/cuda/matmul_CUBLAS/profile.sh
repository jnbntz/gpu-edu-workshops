#!/bin/bash

bin=matmul_CUBLAS

nvprof --output-profile $bin.timeline ./x.$bin
#nvprof --analysis-metrics -o $bin.analysis ./x.$bin
#nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency,shared_replay_overhead -o $bin.metrics ./x.$bin

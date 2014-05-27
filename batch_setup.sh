#!/bin/bash -x

sed -i '/#BATCHARGS/ r cudascript' runit.nvidia-smi
sed -i '/#BATCHARGS/ r cudascript' runit.query
sed -i '/#BATCHARGS/ r cudascript' runit.matmul
sed -i '/#BATCHARGS/ r cudascript' runit.bandwidth

find ./exercises/cuda/ -type f -exec sed -i '/\#BATCHARGS/ r cudascript' {} \;
find ./exercise_solutions/cuda/ -type f -exec sed -i '/\#BATCHARGS/ r cudascript' {} \;

find ./exercises/openacc/ -type f -exec sed -i '/\#BATCHARGS/ r openaccscript' {} \;
find ./exercise_solutions/openacc/ -type f -exec sed -i '/\#BATCHARGS/ r openaccscript' {} \;

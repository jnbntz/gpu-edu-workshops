#!/bin/bash

set -x

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

cc -o mnist mnist.c

./mnist -9 -l t10k-labels-idx1-ubyte -i t10k-images-idx3-ubyte > t10k-images.txt 2> t10k-labels.txt
./mnist -9 -l train-labels-idx1-ubyte -i train-images-idx3-ubyte > train-images.txt 2> train-labels.txt

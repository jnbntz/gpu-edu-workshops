CC=gcc
NVCC=nvcc

include ../make.common

all: x.train x.porterStemmer

x.train: main.o aux.o
	$(NVCC) $(ARCH) -o x.train main.o aux.o -L/usr/lib64/atlas -lcblas -latlas -lcublas

main.o: main.cu headers.h kernels.h
	$(NVCC) $(ARCH) -c main.cu -DDEBUG

aux.o: aux.cu headers.h kernels.h
	$(NVCC) $(ARCH) -c aux.cu -DDEBUG

x.porterStemmer: porterStemmer.o
	$(CC) -o x.porterStemmer porterStemmer.o

porterStemmer.o: porterStemmer.c
	$(CC) -c porterStemmer.c -o porterStemmer.o

clean:
	rm -rf *.o
	rm -rf x.*

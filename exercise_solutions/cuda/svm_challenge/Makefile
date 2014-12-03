CC=gcc
NVCC=nvcc

include ../make.common

all: x.train x.porterStemmer

x.train: main.o aux.o
	$(NVCC) $(ARCH) -o x.train main.o aux.o

main.o: main.cu
	$(NVCC) $(ARCH) -c main.cu

aux.o: aux.cu
	$(NVCC) $(ARCH) -c aux.cu

x.porterStemmer: porterStemmer.o
	$(CC) -o x.porterStemmer porterStemmer.o

porterStemmer.o: porterStemmer.c
	$(CC) -c porterStemmer.c -o porterStemmer.o

clean:
	rm -rf *.o
	rm -rf x.*

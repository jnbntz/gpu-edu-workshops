#
#  Copyright 2017 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

CC=gcc
NVCC=nvcc
NVOPTS=-Xcompiler=-O3 -lineinfo $(ARCH) -DDEBUG
LIBS=-L. -lcblas -latlas -lcublas
INC=-I.

include ../make.common

all: x.nn

x.nn: main.o auxiliary.o
	$(NVCC) $(NVOPTS) -o x.nn main.o auxiliary.o $(LIBS)

main.o: main.cu headers.h
	$(NVCC) $(NVOPTS) -c main.cu $(INC)

auxiliary.o: auxiliary.cu headers.h
	$(NVCC) $(NVOPTS) -c auxiliary.cu $(INC)

clean:
	rm -rf *.o
	rm -rf x.*

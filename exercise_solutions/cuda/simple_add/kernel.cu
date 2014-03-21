/*
 *  Copyright 2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define CUDACHECK( ans ) { gpuAssert( (ans), __FILE__, __LINE__); }
inline void gpuAssert( cudaError_t code, char *file, int line, 
  bool abort=true )
{
    fprintf( stderr, "GPUassert: %s %s %s\n", cudaGetErrorString(code), file, 
      line );
  if( code != cudaSuccess )
  {
    fprintf( stderr, "GPUassert: %s %s %s\n", cudaGetErrorString(code), file, 
      line );
    if( abort ) exit( code );
  } /* end if */
} /* end gpuAssert */

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof( int );

	/* allocate space for device copies of a, b, c */

	CUDACHECK( cudaMalloc( (void **) &d_a, size ) );
	CUDACHECK( cudaMalloc( (void **) &d_b, size ) );
	CUDACHECK( cudaMalloc( (void **) &d_c, size ) );

	/* setup initial values */

	a = 2;
	b = 7;
	c = -99;

	/* copy inputs to device */

	CUDACHECK( cudaMemcpy( d_a, &a, size, cudaMemcpyHostToDevice ) );
	CUDACHECK( cudaMemcpy( d_b, &b, size, cudaMemcpyHostToDevice ) );

	/* launch the kernel on the GPU */

	add<<< 1, 1 >>>( d_a, d_b, d_c );
        CUDACHECK( cudaPeekAtLastError() );

	/* copy result back to host */

	CUDACHECK( cudaMemcpy( &c, d_c, size, cudaMemcpyDeviceToHost ) );

	printf("value of c after kernel is %d\n",c);

	/* clean up */

	CUDACHECK( cudaFree( d_a ) );
	CUDACHECK( cudaFree( d_b ) );
	CUDACHECK( cudaFree( d_c ) );

        cudaDeviceReset();
	
	return 0;
} /* end main */

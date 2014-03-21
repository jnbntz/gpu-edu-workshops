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
#include <stdio.h>
#include <math.h>

/* definitions of threadblock size in X and Y directions */

#define THREAD_X 16
#define THREAD_Y 16

/* definition of matrix linear dimension */

#define SIZE 1024

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for naive matrix transpose */

__global__ void naive_cuda_transpose( const int m, double const * const a, double *c )
{
	/* insert code to calculate global row and column of the matrix */
	const int myRow = FIXME
	const int myCol = FIXME

	if( myRow < m && myCol < m )
	{
		/* insert the indices for accessing the A and C matrices to execute the transpose */
             c[FIXME] = a[FIXME];
	} /* end if */
	return;
} /* end naive_cuda_transpose */

void host_transpose( const int m, double const * const a, double *c )
{
	
/* 
 *  naive matrix transpose on CPU goes here.
 */
 
 for( int j = 0; j < m; j++ )
	{
		for( int i = 0; i < m; i++ )
		{
		    c[INDX(i,j,m)] = a[INDX(j,i,m)];
		} /* end for i */
	} /* end for j */

} /* end host_dgemm */

int main( int argc, char *argv[] )
{

    int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

/* declaring pointers for array */

    double *h_a, *h_c;
    double *d_a, *d_c;
 
    size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* allocating host memory */

    h_a = (double *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc h_a\n");
      return 911;
    }

    h_c = (double *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc h_c\n");
      return 911;
    }

/* allocating device memory */

    cudaMalloc( (void**) &d_a, numbytes );
    cudaMalloc( (void**) &d_c, numbytes );

/* set result matrices to zero */

    memset( h_c, 0, numbytes );
    cudaMemset( d_c, 0, numbytes );

    fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
       (double) numbytes / 1000000.0 );

/* initialize input matrix with random value */

    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    }

/* copy input matrix from host to device */

    cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice );

/* create and start timer */

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

/* call naive cpu transpose function */

    host_transpose( size, h_a, h_c );

/* stop CPU timer */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

/* print CPU timing information */

    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GB/s\n", 
      8.0 * 2.0 * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* setup threadblock size and grid sizes */

    dim3 threads( THREAD_X, THREAD_Y, 1 );

	/* insert code for proper grids in X and Y directions */
    dim3 blocks( FIXME, FIXME, 1 );

/* start timers */
    cudaEventRecord( start, 0 );

/* call naive GPU transpose kernel */

    naive_cuda_transpose<<< blocks, threads >>>( size, d_a, d_c );

/* stop the timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

/* print GPU timing information */

    fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GB/s\n", 
      8.0 * 2.0 * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* copy data from device to host */

    cudaMemset( h_a, 0, numbytes );
    cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost );

/* compare GPU to CPU for correctness */

	for( int j = 0; j < size; j++ )
	{
		for( int i = 0; i < size; i++ )
		{
		    if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
                    {
                      printf("Error in element %d,%d\n", i,j );
                      printf("Host %f, device %d\n",h_c[INDX(i,j,size)],
                                                    h_a[INDX(i,j,size)]);
                    }
		} /* end for i */
	} /* end for j */

/* free the memory */

    free( h_a );
    free( h_c );
    cudaFree( d_a );
    cudaFree( h_a );

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

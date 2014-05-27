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
#include "cublas_v2.h"

#include <stdio.h>

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* matrix size and thread dimensions */

#define SIZE 1024

/* define blocksize X and blocksize Y and blocksize K */

#define TX 16 // Thread block size, x dimension
#define TY 16 // Thread block size, y dimension
#define BK 16 // square block of K size

__global__ void GPU_shmem2(const int m, double const * const a, double const * const b, double *c )
{

/* setup some constanst for later use */

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int iby = blockIdx.y * TY;
	const int ibx = blockIdx.x * TX;

/* shared memory arrays for A and B */
	/* insert code for proper size of shared memory segments */
	__shared__ double as[ FIXME ][ FIXME ];
	__shared__ double bs[ FIXME ][ FIXME ];
	
/* space for C to be held in register */

	double c_tmp = 0.0 ;

	/* calculate my thread's initial offset into the A and B matrices */
	/* insert code for proper offset into A and B matrices */
	int aoff = INDX( FIXME, FIXME, m );
	int boff = INDX( FIXME, FIXME, m );

	/* main loop over blocks of K */

	for( int Kblock = 0; Kblock < m; Kblock+=BK )
	{

		/* read block of A into shared memory */
		/* insert code for proper indices */
			as[ FIXME ][ FIXME ] = a[ aoff ];

		/* read block of B into shared memory */
		/* insert code for proper indices */
			bs[ FIXME ][ FIXME ] = b[ boff ];

		/* increment A and B offsets for next round of data reads */

		boff += BK;
		aoff += m * BK;

		/* loop to perform the matmult on the blocks */
		/* insert code for proper indices */

#pragma unroll
		for( int k = 0 ; k < BK ; k++ )
		{
			/* insert code for proper indices into the a and b shared matrices */
			c_tmp += as[ FIXME ][ FIXME ] * bs[ FIXME ][ FIXME ];
		}


	} /* end for Kblock */

	/* set C to its proper index into the C matrix */

	int coff = INDX( ibx + tx, iby + ty, m );

	/* write results to the C matrix */

	c[ coff ] = c_tmp;
 
} /* end GPU_shmem2 */


int main( int argc, char *argv[] )
{

    const int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

    double *h_a, *h_b, *h_c, *h_c1;
    double *d_a, *d_b, *d_c;
 
    size_t numbytes = (size_t ) size * (size_t ) size * sizeof( double );

    h_a = (double *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_b = (double *) malloc( numbytes );
    if( h_b == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_c = (double *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

	h_c1 = (double *) malloc( numbytes );
    if( h_c1 == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

	/* zero out the host memory for C matrices */

    memset( h_c, 0, numbytes );
    memset( h_c1, 0, numbytes );

    fprintf( stdout, "Total memory required is %lf MB\n", 
       3.0 * (double) numbytes / 1000000.0 );

	/* initialize the A and B matrices */

    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
      h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    }

	/* allocate a, b, c in gpu memory */

    cudaMalloc( (void **)&d_a, numbytes );
    cudaMalloc( (void **)&d_b, numbytes );
    cudaMalloc( (void **)&d_c, numbytes );
	
	/* copy a and b to device */

	cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice );

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    double alpha = 1.0;
    double beta  = 0.0;

	/* start timers */

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

	/* call CUBLAS dgemm */

cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha, 
                 d_a, size,
                 d_b, size,
                 &beta,
                 d_c, size );

	/* stop timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

	/* print GPU CUBLAS timing information */

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
    
	/* copy C from device to host for error checking */

    cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost );

	/* reset C on device to zero */

	cudaMemset( d_c, 0, numbytes );

	/* setup grid and block sizes */

	dim3 threads( TX, TY, 1 );
	dim3 blocks( size / TX, size / TY, 1 );

	/* start timers */

	cudaEventRecord( start, 0 );

	/* call GPU_naive */

	GPU_shmem2<<< blocks, threads >>> ( size, d_a, d_b, d_c );

	/* stop timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

	/* print data for GPU naive */

    fprintf(stdout, "Total time GPU SHMEM is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
                  
	/* copy C back to host */
	
	cudaMemcpy( h_c1, d_c, numbytes, cudaMemcpyDeviceToHost );

    cublasDestroy( handle );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	/* check CUBLAS versus GPU NAIVE numerical results */

    double temp = 0.0;

    for( int i = 0; i < size * size; i++ )
    {
      temp += ( h_c[i] - h_c1[i] ) * ( h_c[i] - h_c1[i] );
    } /* end for */

    printf("error is %f\n",temp);
    if( temp > 10 ) printf("FAIL\n");
    else printf("PASS\n");

	/* cleanup */

    cudaFree( d_a );
    cudaFree( d_b );
	cudaFree( d_c );

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_c1 );

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

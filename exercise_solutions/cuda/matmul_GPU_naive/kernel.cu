/*
 *  Copyright 2017 NVIDIA Corporation
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

#include <stdio.h>
#include "cublas_v2.h"
#include "../debug.h"

typedef float floatType_t;

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* matrix size and thread dimensions */

#define SIZE 1024
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

/* naive GPU kernel where each element of C is computed by a single thread */

__global__ void GPU_naive( const int m, floatType_t const * const a, 
     floatType_t const * const b, floatType_t * const c )
{

/* determine my threads's row and col indices in the global C matrix */

  const int myrow = blockDim.x * blockIdx.x + threadIdx.x;
  const int mycol = blockDim.y * blockIdx.y + threadIdx.y;

/* if my row and col are in the C matrix, then calculate that value of C */

  if( myrow < m && mycol < m )
  {
    register floatType_t temp = 0.0;

    for( int k = 0; k < m; k++ ) 
      temp += a[INDX( myrow, k, m )] * b[INDX( k, mycol, m )];

    c[INDX( myrow, mycol, m )] = temp;
  } /* end if */

	return;
} /* end GPU_naive */

int main( int argc, char *argv[] )
{

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  const int size = SIZE;

  fprintf(stdout, "Matrix size is %d\n",size);

  floatType_t *h_a, *h_b, *h_c, *h_c1;
  floatType_t *d_a, *d_b, *d_c;
 
  size_t numbytes = (size_t ) size * (size_t ) size * sizeof( floatType_t );

  h_a = (floatType_t *) malloc( numbytes );
  if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_b = (floatType_t *) malloc( numbytes );
  if( h_b == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_c = (floatType_t *) malloc( numbytes );
  if( h_c == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_c1 = (floatType_t *) malloc( numbytes );
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

  checkCUDA( cudaMalloc( (void **)&d_a, numbytes ) );
  checkCUDA( cudaMalloc( (void **)&d_b, numbytes ) );
  checkCUDA( cudaMalloc( (void **)&d_c, numbytes ) );
	
/* copy a and b to device */

  checkCUDA( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice ) );

  cublasHandle_t handle;
  checkCUBLAS( cublasCreate( &handle ) );

  floatType_t alpha = 1.0;
  floatType_t beta  = 0.0;

/* start timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call CUBLAS dgemm */

  if( sizeof( floatType_t ) == 4 ) 
  {
  checkCUBLAS( 
  cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               size, size, size,
               (float *)&alpha, 
               (float *)d_a, size,
               (float *)d_b, size,
               (float *)&beta,
               (float *)d_c, size )
              );
  } /* end if */
  else
  {
  checkCUBLAS( 
  cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               size, size, size,
               (double *)&alpha, 
               (double *)d_a, size,
               (double *)d_b, size,
               (double *)&beta,
               (double *)d_c, size )
              );
  } /* end else */

/* stop timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU CUBLAS timing information */

  fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", 
    2.0 * (double) size * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
    
/* copy C from device to host for error checking */

  checkCUDA( cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* reset C on device to zero */

  checkCUDA( cudaMemset( d_c, 0, numbytes ) );

/* setup grid and block sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( size / THREADS_PER_BLOCK_X + 1, 
               size / THREADS_PER_BLOCK_Y + 1, 1 );

/* start timers */

  checkCUDA( cudaEventRecord( start, 0 ) );

/* call GPU_naive */

  GPU_naive<<< blocks, threads >>> ( size, d_a, d_b, d_c );
  checkKERNEL()

/* stop timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print data for GPU naive */

  fprintf(stdout, "Total time GPU NAIVE is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", 
    2.0 * (double) size * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
                  
/* copy C back to host */
	
  checkCUDA( cudaMemcpy( h_c1, d_c, numbytes, cudaMemcpyDeviceToHost ) );

  checkCUBLAS( cublasDestroy( handle ) );
  checkCUDA( cudaEventDestroy( start ) );
  checkCUDA( cudaEventDestroy( stop ) );

/* check CUBLAS versus GPU NAIVE numerical results */

  double temp = 0.0;

  for( int i = 0; i < size * size; i++ )
  {
     temp = max( temp, abs( (double)h_c[i] - (double)h_c1[i] )/
                      abs((double)h_c[i]) );
  } /* end for */
  printf("Maximum error is %e percent \n",temp*100.0);
  if( temp > 0.001 ) printf("FAIL\n");
  else printf("PASS\n");

/* cleanup */

  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_b ) );
  checkCUDA( cudaFree( d_c ) );

  free( h_a );
  free( h_b );
  free( h_c );
  free( h_c1 );

  checkCUDA( cudaDeviceReset() );

  return 0;
}

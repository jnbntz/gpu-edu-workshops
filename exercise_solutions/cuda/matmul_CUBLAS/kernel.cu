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

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

#include <stdio.h>
#include "cublas_v2.h"

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

#define SIZE 1024

void host_dgemm( int m, int n, int k, double *a, double *b, double *c )
{

  for( int j = 0; j < n; j++ )
  {
    for( int i = 0; i < m; i++ )
    {
      for( int koff = 0; koff < k; koff++ )
      {
	c[INDX(i, j, m)] += a[INDX( i, koff, m )] * b[INDX( koff, j, n )];
      } /* end for i */
    } /* end jb */
  } /* end for j */

} /* end host_dgemm */

int main( int argc, char *argv[] )
{

    const int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

    double *h_a, *h_b, *h_c, *h_cdef;
    double *d_a, *d_b, *d_c;
 
    size_t numbytes = size * size * sizeof( double );

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

    h_cdef = (double *) malloc( numbytes );
    if( h_cdef == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    memset( h_c, 0, numbytes );
    memset( h_cdef, 0, numbytes );

    fprintf( stdout, "Total memory required is %lf MB\n", 
       3.0 * (double) numbytes / 1000000.0 );

    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
      h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    }

    CUDA_CALL( cudaMalloc( (void **)&d_a, numbytes ) );
    CUDA_CALL( cudaMalloc( (void **)&d_b, numbytes ) );
    CUDA_CALL( cudaMalloc( (void **)&d_c, numbytes ) );

    cudaEvent_t start, stop;
    CUDA_CALL( cudaEventCreate( &start ) );
    CUDA_CALL( cudaEventCreate( &stop ) );


    CUDA_CALL( cudaEventRecord( start, 0 ) );

    host_dgemm( size, size, size, h_a, h_b, h_cdef );

    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    float elapsedTime;
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );


    CUDA_CALL( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice ) );

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    double alpha = 1.0;
    double beta  = 0.0;

    CUDA_CALL( cudaEventRecord( start, 0 ) );

    cublasSetStream( handle, 0 );

    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha, 
                 d_a, size,
                 d_b, size,
                 &beta,
                 d_c, size );


    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", 
            elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
                  
    CUDA_CALL( cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost ) );

    cublasDestroy( handle );
    CUDA_CALL( cudaEventDestroy( start ) );
    CUDA_CALL( cudaEventDestroy( stop ) );

    double temp = 0.0;

    for( int i = 0; i < size * size; i++ )
    {
      temp += ( h_c[i] - h_cdef[i] ) * ( h_c[i] - h_cdef[i] );
    } /* end for */
    printf("error is %f\n",temp);
    if( temp > 10 ) printf("FAIL\n");
    else printf("PASS\n");

    CUDA_CALL( cudaFree( d_a ) );
    CUDA_CALL( cudaFree( d_b ) );
    CUDA_CALL( cudaFree( d_c ) );

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_cdef );

    CUDA_CALL( cudaDeviceReset() );
    return 0;
}

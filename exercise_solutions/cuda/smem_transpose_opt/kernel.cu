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

#include <stdio.h>
#include <math.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif

/* definitions of threadblock size in X and Y directions */

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 4
#define ELEMENTS_PER_THREAD 8

/* definition of matrix linear dimension */

#define SIZE 4096

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for shared memory matrix transpose */

__global__ void smem_cuda_transpose_opt( const int m,  
                                         const double *a, 
                                         double *c )
{
	
/* declare a shared memory array */

  __shared__ double smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y*ELEMENTS_PER_THREAD+1];
	
  {
  int aoff = INDX( blockDim.x * blockIdx.x + threadIdx.x, 
                   blockDim.x * blockIdx.y + threadIdx.y, m );

  for( int i = 0; i < ELEMENTS_PER_THREAD; i++ ) 
    smemArray[threadIdx.x][threadIdx.y + i*blockDim.y] = 
      a[ aoff + i*blockDim.y*m ];
  }
/* synchronize */
  __syncthreads();

/* write the result */
  {
  int coff = INDX( blockDim.x * blockIdx.y + threadIdx.x, 
                   blockDim.x * blockIdx.x + threadIdx.y, m );

#pragma unroll
  for( int i = 0; i < ELEMENTS_PER_THREAD; i++ )
    c[coff + i*blockDim.y*m] = 
      smemArray[threadIdx.y + i*blockDim.y][threadIdx.x];
  }
  return;

} /* end naive_cuda_transpose */

void host_transpose( const int m, double const * const a, double *c )
{
	
/* 
 *  naive matrix transpose goes here.
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

  CUDA_CALL( cudaMalloc( (void**) &d_a, numbytes ) );
  CUDA_CALL( cudaMalloc( (void**) &d_c, numbytes ) );

/* set result matrices to zero */

  memset( h_c, 0, numbytes );
  CUDA_CALL( cudaMemset( d_c, 0, numbytes ) );

  fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
     (double) numbytes / 1000000.0 );

/* initialize input matrix with random value */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
//    h_a[i] = double( i );
  } /* end for */

/* copy input matrix from host to device */

  CUDA_CALL( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );

/* create and start timer */

  cudaEvent_t start, stop;
  CUDA_CALL( cudaEventCreate( &start ) );
  CUDA_CALL( cudaEventCreate( &stop ) );
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* call naive cpu transpose function */

  host_transpose( size, h_a, h_c );

/* stop CPU timer */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  float elapsedTime;
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print CPU timing information */

  fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* setup threadblock size and grid sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( ( size / THREADS_PER_BLOCK_X ) , 
               ( size / ( THREADS_PER_BLOCK_Y * ELEMENTS_PER_THREAD ) ) , 1 );

  CUDA_CALL( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );

/* start timers */
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* call naive GPU transpose kernel */

  smem_cuda_transpose_opt<<< blocks, threads >>>( size, d_a, d_c );
  CUDA_CHECK();
  CUDA_CALL( cudaDeviceSynchronize() );

/* stop the timers */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU timing information */

  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* copy data from device to host */

  CUDA_CALL( cudaMemset( d_a, 0, numbytes ) );
  CUDA_CALL( cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* compare GPU to CPU for correctness */

  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
      {
        printf("Error in element %d,%d\n", i,j );
        printf("Host %f, device %f\n",h_c[INDX(i,j,size)],
                                      h_a[INDX(i,j,size)]);
      }
    } /* end for i */
  } /* end for j */

/* free the memory */

  free( h_a );
  free( h_c );
  CUDA_CALL( cudaFree( d_a ) );
  CUDA_CALL( cudaFree( d_c ) );

  CUDA_CALL( cudaDeviceReset() );

  return 0;
}

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

//#include <stdio.h>
#include <iostream>
#include "../debug.h"

using namespace std;

/* definitions of threadblock size in X and Y directions */

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* definition of matrix linear dimension */

#define SIZE 4096

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for naive matrix transpose */

__global__ void naive_cuda_transpose( const int m, 
                                      const double * const a, 
                                      double * const c )
{
  const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
  const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

  if( myRow < m && myCol < m )
  {
    c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
  } /* end if */
  return;

} /* end naive_cuda_transpose */

void host_transpose( const int m, const double * const a, double *c )
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

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  cout << "Using GPU " << dev << ": " << deviceProp.name << endl;

  int size = SIZE;

  cout << "Matrix size is " << size << endl;

/* declaring pointers for array */

  double *h_a, *h_c;
  double *d_a, *d_c;
 
  size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* allocating host memory */

  h_a = (double *) malloc( numbytes );
  if( h_a == NULL )
  {
	cerr << "Error in host Malloc h_a " << endl;
    return 911;
  }

  h_c = (double *) malloc( numbytes );
  if( h_c == NULL )
  {
	cerr << "Error in host Malloc h_c " << endl;
    return 911;
  }

/* allocating device memory */

  checkCUDA( cudaMalloc( (void**) &d_a, numbytes ) );
  checkCUDA( cudaMalloc( (void**) &d_c, numbytes ) );

/* set result matrices to zero */

  memset( h_c, 0, numbytes );
  checkCUDA( cudaMemset( d_c, 0, numbytes ) );

  cout << "Total memory required per matrix is " << (double) numbytes / 1000000.0 << endl;

/* initialize input matrix with random value */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  }

/* copy input matrix from host to device */

  checkCUDA( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );

/* create and start timer */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call naive cpu transpose function */

  host_transpose( size, h_a, h_c );

/* stop CPU timer */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print CPU timing information */

  cout << "Total time CPU is " << elapsedTime / 1000.0f << " sec" << endl;
  cout << "Performance is " << 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 << endl ; 
//  fprintf(stdout, "Performance is %f GB/s\n", 
 //   8.0 * 2.0 * (double) size * (double) size / 
  //  ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* setup threadblock size and grid sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( ( size / THREADS_PER_BLOCK_X ) + 1, 
               ( size / THREADS_PER_BLOCK_Y ) + 1, 1 );

/* start timers */
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call naive GPU transpose kernel */

  naive_cuda_transpose<<< blocks, threads >>>( size, d_a, d_c );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU timing information */

//  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
 // fprintf(stdout, "Performance is %f GB/s\n", 
  //  8.0 * 2.0 * (double) size * (double) size / 
   // ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

  cout << "Total time GPU is " << elapsedTime / 1000.0f << " sec" << endl;
  cout << "Performance is " << 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 << endl ; 

/* copy data from device to host */

  checkCUDA( cudaMemset( d_a, 0, numbytes ) );
  checkCUDA( cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* compare GPU to CPU for correctness */

  int success = 1;

  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
      {
		cerr << "Error in element " << i << "," << j << endl;
		cerr << "Host " << h_c[INDX(i,j,size)] << ", device " << h_a[INDX(i,j,size)] << endl;
        success = 0;
        break;
      } /* end fi */
    } /* end for i */
  } /* end for j */

  if( success == 1 ) cout << "PASS" << endl;
  else               cout << "FAIL" << endl;

/* free the memory */
  free( h_a );
  free( h_c );
  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_c ) );
  checkCUDA( cudaDeviceReset() );

  return 0;
} /* end main */

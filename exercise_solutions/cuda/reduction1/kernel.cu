/*
 *  Copyright 2015 NVIDIA Corporation
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
#include "../debug.h"

#define N ( 8192 * 8192 )
#define THREADS_PER_BLOCK 512

typedef float floatType_t;

__global__ void sumReduction(int n, floatType_t *in, floatType_t *out)
{
/* calculate global index in the array */
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
/* return if my global index is larger than the array size */
  if( globalIndex >= n ) return;

  floatType_t result = 0.0;

  for( int i = 0; i < N; i++ ) 
  { 
    result += in[i];
  } /* end for */

  *out = result;
  return;

}

int main()
{
  floatType_t *h_in, h_out, good_out;
  floatType_t *d_in, *d_out;
  int memBytes = N * sizeof( floatType_t );

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* allocate space for device copies of in, out */

  checkCUDA( cudaMalloc( &d_in, memBytes ) );
  checkCUDA( cudaMalloc( &d_out, sizeof(floatType_t) ) );

/* allocate space for host copies of in, out and setup input values */

  h_in = (floatType_t *)malloc( memBytes );

  for( int i = 0; i < N; i++ )
  {
   h_in[i] = floatType_t( rand() ) / ( floatType_t (RAND_MAX) + 1.0 );
  }

  h_out      = 0.0;
  good_out   = 0.0;

/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_in, h_in, memBytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemset( d_out, 0, sizeof(floatType_t) ) );

/* calculate block and grid sizes */

  dim3 threads( THREADS_PER_BLOCK, 1, 1);
  dim3 blocks( (N / threads.x) + 1, 1, 1);

/* start the timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* launch the kernel on the GPU */

  sumReduction<<< 1, 1 >>>( N, d_in, d_out );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  printf("Total time for %d elements was %f ms\n", N, elapsedTime );
  printf("Performance is %f GB/s for %f GB of data.\n",
    sizeof(floatType_t) * (double) N /
    ( (double) elapsedTime / 1000.0 ) * 1.e-9,
    sizeof(floatType_t) * (double) N * 1.e-9 );


/* copy result back to host */

  checkCUDA( cudaMemcpy( &h_out, d_out, sizeof(floatType_t), 
    cudaMemcpyDeviceToHost ) );

  for( int i = 0; i < N; i++ )
  {
    good_out += h_in[i];
  } /* end for */

  floatType_t diff = abs( good_out - h_out );

//  printf("diff is %e\n",diff);
//  printf("Sum is %e\n", h_out);

  if( diff / h_out < 0.001 ) printf("PASS\n");
  else                       printf("FAIL\n");

/* clean up */

  free(h_in);
  checkCUDA( cudaFree( d_in ) );
  checkCUDA( cudaFree( d_out ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

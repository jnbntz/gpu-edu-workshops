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
#include "../debug.h"

#define N ( 1024 * 1024 )
#define RADIUS 5
#define THREADS_PER_BLOCK 512

__global__ void stencil_1d(int n, double *in, double *out)
{
/* calculate global index in the array */
/* insert code to calculate global index in the array using block
   and thread built-in variables */
  int globalIndex = FIXME
	
/* return if my global index is larger than the array size */
  if( globalIndex >= n ) return;

/* code to handle the boundary conditions */
  if( globalIndex < RADIUS || globalIndex >= (n - RADIUS) ) 
  {
    out[globalIndex] = (double) globalIndex * ( (double)RADIUS*2 + 1) ;
    return;
  } /* end if */

  double result = 0.0;
	
  for( int i = globalIndex-(RADIUS); i <= globalIndex+(RADIUS); i++ ) 
  {
/* add the required elements from the array "in" to the temporary 
   variable "result */
    result += FIXME
  }

  out[globalIndex] = result;
  return;

}

int main()
{
/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  double *in, *out;
  double *d_in, *d_out;
  int size = N * sizeof( double );

/* allocate space for device copies of in, out */

  checkCUDA( cudaMalloc( (void **) &d_in, size ) );
  checkCUDA( cudaMalloc( (void **) &d_out, size ) );

/* allocate space for host copies of in, out and setup input values */

  in = (double *)malloc( size );
  out = (double *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    in[i] = (double) i;
    out[i] = 0;
  }

/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_in, in, size, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemset( d_out, 0, size ) );

/* calculate block and grid sizes */

  dim3 threads( THREADS_PER_BLOCK, 1, 1);
/* insert code for proper number of blocks in X dimension */
  dim3 blocks( FIXME, 1, 1);

/* start the timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* launch the kernel on the GPU */

  stencil_1d<<< blocks, threads >>>( N, d_in, d_out );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  printf("Total time for %d elements was %f ms\n", N, elapsedTime );

/* copy result back to host */

  checkCUDA( cudaMemcpy( out, d_out, size, cudaMemcpyDeviceToHost ) );

  int success = 1;

  for( int i = 0; i < N; i++ )
  {
    if( in[i]*( (double)RADIUS*2+1 ) != out[i] )
    {
      printf("error in element %d in = %f out %f\n",i,in[i],out[i] );
      success = 0;
      break;
    } /* end if */
  } /* end for */

  if( success == 1 ) printf("PASS\n");
  else               printf("FAIL\n");

/* clean up */

  free(in);
  free(out);
  checkCUDA( cudaFree( d_in ) );
  checkCUDA( cudaFree( d_out ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

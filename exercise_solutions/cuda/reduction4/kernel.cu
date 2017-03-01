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

#define N ( 1 << 27 )
#define THREADS_PER_BLOCK 128

#define FLOATTYPE_T float

__global__ void sumReduction(int n, FLOATTYPE_T *in, FLOATTYPE_T *out)
{

  __shared__ FLOATTYPE_T sArray[THREADS_PER_BLOCK];

/* calculate global index in the array */
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
/* zero out the smem array */
  sArray[threadIdx.x] = 0.0;

/* grid stride handling case where array is larger than number of threads
 * launched
 * Loop over the grid stride so that each thread adds up its relevant 
 * elements of the array and saves them to SMEM
 */

  for( int i = globalIndex; i < n; i += blockDim.x * gridDim.x )
  {
    sArray[threadIdx.x] += in[i];
  } /* end for */
  
  __syncthreads();

/* do the final reduction in SMEM */
  for( int i = blockDim.x/2; i > 0; i = i / 2 )
  {
    if( threadIdx.x < i )
    {
      sArray[threadIdx.x] += sArray[threadIdx.x + i];
    } /* end if */
    __syncthreads();
  } /* end for */

/* thread0 of each threadblock writes the result to global memory */
  if( threadIdx.x == 0 ) out[blockIdx.x] = sArray[0]; 

  return;

}

int main()
{
  FLOATTYPE_T *h_in, h_sum, cpu_sum;
  FLOATTYPE_T *d_in, *d_sum, *d_tempArray;
  int size = N;
  int memBytes = size * sizeof( FLOATTYPE_T );
  int tempArraySize = 32768;

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* allocate space for device copies of in, out */

  checkCUDA( cudaMalloc( &d_in, memBytes ) );
  checkCUDA( cudaMalloc( &d_sum, sizeof(FLOATTYPE_T) ) );
  checkCUDA( cudaMalloc( &d_tempArray, tempArraySize * sizeof(FLOATTYPE_T) ) );

/* allocate space for host copies of in, out and setup input values */

  h_in = (FLOATTYPE_T *)malloc( memBytes );

  for( int i = 0; i < size; i++ )
  {
    h_in[i] = FLOATTYPE_T( rand() ) / ( FLOATTYPE_T (RAND_MAX) + 1.0 );
    if( i % 2 == 0 ) h_in[i] = -h_in[i];
  }

  h_sum      = 0.0;
  cpu_sum   = 0.0;

/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_in, h_in, memBytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemset( d_sum, 0, sizeof(FLOATTYPE_T) ) );
  checkCUDA( cudaMemset( d_tempArray, 0, 
    tempArraySize * sizeof(FLOATTYPE_T) ) );

/* calculate block and grid sizes */

  dim3 threads1( THREADS_PER_BLOCK, 1, 1 );
  
  int blk = min( (size / threads1.x), tempArraySize );
  dim3 blocks( blk, 1, 1);

  dim3 threads2( min(blocks.x,threads1.x), 1, 1 );

/* start the timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* launch the kernel on the GPU */

  sumReduction<<< blocks, threads1 >>>( size, d_in,  d_tempArray );
  checkKERNEL()
  sumReduction<<<      1, threads2 >>>( blocks.x, d_tempArray, d_sum );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  printf("Total elements is %d, %f GB\n", size, sizeof(FLOATTYPE_T)*
    (double)size * 1.e-9 );
  printf("GPU total time is %f ms, bandwidth %f GB/s\n", elapsedTime,
    sizeof(FLOATTYPE_T) * (double) size /
    ( (double) elapsedTime / 1000.0 ) * 1.e-9);

/* copy result back to host */

  checkCUDA( cudaMemcpy( &h_sum, d_sum, sizeof(FLOATTYPE_T), 
    cudaMemcpyDeviceToHost ) );

  checkCUDA( cudaEventRecord( start, 0 ) );

  for( int i = 0; i < size; i++ )
  {
    cpu_sum += h_in[i];
  } /* end for */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf("CPU total time is %f ms, bandwidth %f GB/s\n", elapsedTime,
    sizeof(FLOATTYPE_T) * (double) size /
    ( (double) elapsedTime / 1000.0 ) * 1.e-9);


  FLOATTYPE_T diff = abs( cpu_sum - h_sum );

  if( diff / abs(h_sum) < 0.001 ) printf("PASS\n");
  else
  {                       
    printf("FAIL\n");
    printf("Error is %f\n", diff / h_sum );
    printf("GPU result is %f, CPU result is %f\n",h_sum, cpu_sum );
  } /* end else */

/* clean up */

  free(h_in);
  checkCUDA( cudaFree( d_in ) );
  checkCUDA( cudaFree( d_sum ) );
  checkCUDA( cudaFree( d_tempArray ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

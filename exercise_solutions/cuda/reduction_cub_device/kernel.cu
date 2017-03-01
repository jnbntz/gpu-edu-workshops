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
#include <cub/cub.cuh>
#include "../debug.h"

#define N ( 1 << 27 )
#define FLOATTYPE_T float

int main()
{
  FLOATTYPE_T *h_in, h_out, good_out;
  FLOATTYPE_T *d_in, *d_out, *d_tempArray;
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
  checkCUDA( cudaMalloc( &d_out, sizeof(FLOATTYPE_T) ) );
  checkCUDA( cudaMalloc( &d_tempArray, tempArraySize * sizeof(FLOATTYPE_T) ) );

/* allocate space for host copies of in, out and setup input values */

  h_in = (FLOATTYPE_T *)malloc( memBytes );

  for( int i = 0; i < size; i++ )
  {
    h_in[i] = FLOATTYPE_T( rand() ) / ( FLOATTYPE_T (RAND_MAX) + 1.0 );
    if( i % 2 == 0 ) h_in[i] = -h_in[i];
  }

  h_out      = 0.0;
  good_out   = 0.0;

/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_in, h_in, memBytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemset( d_out, 0, sizeof(FLOATTYPE_T) ) );
  checkCUDA( cudaMemset( d_tempArray, 0, 
    tempArraySize * sizeof(FLOATTYPE_T) ) );

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_in, d_out, 
    size );

  printf("temp storage is %ld\n", temp_storage_bytes );

  checkCUDA( cudaMalloc( &d_temp_storage, temp_storage_bytes ) ); 

/* start the timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* launch the kernel on the GPU */
  cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_in, d_out, 
    size );

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

  checkCUDA( cudaMemcpy( &h_out, d_out, sizeof(FLOATTYPE_T), 
    cudaMemcpyDeviceToHost ) );

  checkCUDA( cudaEventRecord( start, 0 ) );

  for( int i = 0; i < size; i++ )
  {
    good_out += h_in[i];
  } /* end for */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf("CPU total time is %f ms, bandwidth %f GB/s\n", elapsedTime,
    sizeof(FLOATTYPE_T) * (double) size /
    ( (double) elapsedTime / 1000.0 ) * 1.e-9);


  FLOATTYPE_T diff = abs( good_out - h_out );

  if( diff / abs(h_out) < 0.001 ) printf("PASS\n");
  else
  {                       
    printf("FAIL\n");
    printf("Error is %f\n", diff / h_out );
    printf("GPU result is %f, CPU result is %f\n",h_out, good_out );
  } /* end else */

/* clean up */

  free(h_in);
  checkCUDA( cudaFree( d_in ) );
  checkCUDA( cudaFree( d_out ) );
  checkCUDA( cudaFree( d_tempArray ) );
  checkCUDA( cudaFree( d_temp_storage ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

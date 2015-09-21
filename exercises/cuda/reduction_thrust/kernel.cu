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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include "../debug.h"

#define N ( 1 << 27 )
#define FLOATTYPE_T float 

int main(void)
{
  int size = N;

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* create the host array */  
  thrust::host_vector<FLOATTYPE_T> h_vec( size );

/* generate random numbers on the host */
  for( int i = 0; i < size; i++ )
  {
    h_vec[i] = FLOATTYPE_T( rand() ) / ( FLOATTYPE_T (RAND_MAX) + 1.0 );
    if( i % 2 == 0 ) h_vec[i] = -h_vec[i];
  }

/* transfer data to the device */
  thrust::device_vector<FLOATTYPE_T> d_vec = h_vec;

/* create timers */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );

/* reduce data on the device */
  FLOATTYPE_T devResult = thrust::reduce( d_vec.begin(), d_vec.end() );

/* stop timers */
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float GPUelapsedTime;
  cudaEventElapsedTime( &GPUelapsedTime, start, stop );

  GPUelapsedTime /= 1000.0;

/* print GPU timing data */

  printf("Total elements is %d, %f GB\n", size, sizeof(FLOATTYPE_T) * 
    (double)size * 1.e-9);
  printf("GPU total time is %f ms, bandwidth %f GB/s\n", GPUelapsedTime,
    sizeof(FLOATTYPE_T)*(double)size / 
    ( (double)GPUelapsedTime ) * 1.e-9 );

/* start CPU timer */
  cudaEventRecord( start, 0 );

/* reduce data on host */
  FLOATTYPE_T hostResult = thrust::reduce(h_vec.begin(), h_vec.end() );

/* stop timers */
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float CPUelapsedTime;
  cudaEventElapsedTime( &CPUelapsedTime, start, stop );
  CPUelapsedTime /= 1000.0;

/* print CPU timer */

  printf("Total elements is %d, %f GB\n", size, sizeof(FLOATTYPE_T) * 
    (double)size * 1.e-9);
  printf("CPU total time is %f ms, bandwidth %f GB/s\n", CPUelapsedTime,
    sizeof(FLOATTYPE_T)*(double)size / 
    ( (double)CPUelapsedTime ) * 1.e-9 );


  cudaEventDestroy(start);
  cudaEventDestroy(stop);

/* verify the results */

  double diff = abs( devResult - hostResult );

  if( diff / hostResult < 0.001 ) printf("PASS\n");
  else
  {
    printf("FAIL\n");
    printf("Error is %f\n", diff / hostResult );
    printf("GPU result is %f, CPU result is %f\n",devResult, hostResult );
  } /* end else */

  return 0;
} /* end main */


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

#define N ( 1 << 26 )
#define FLOATTYPE_T double

int main(void)
{
  int size = N;

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  // generate random numbers on the host
  thrust::host_vector<FLOATTYPE_T> h_vec( size );
  thrust::generate( h_vec.begin(), h_vec.end(), rand );

  //transfer data to the device
  thrust::device_vector<FLOATTYPE_T> d_vec = h_vec;

  //create timers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );

  //reduce data on the device
  double devResult = thrust::reduce( d_vec.begin(), d_vec.end() );

/* stop timers */
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float GPUelapsedTime;
  cudaEventElapsedTime( &GPUelapsedTime, start, stop );

  GPUelapsedTime /= 1000.0;

  printf("Total elements is %d, %f GB\n", size, sizeof(double) * 
    (double)size * 1.e-9);
  printf("GPU total time is %f ms, bandwidth %f GB/s\n", GPUelapsedTime,
    sizeof(double)*(double)size / 
    ( (double)GPUelapsedTime ) * 1.e-9 );


  cudaEventRecord( start, 0 );

  //reduce data on host
  double hostResult = thrust::reduce(h_vec.begin(), h_vec.end() );

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float CPUelapsedTime;
  cudaEventElapsedTime( &CPUelapsedTime, start, stop );
  CPUelapsedTime /= 1000.0;

  printf("Total elements is %d, %f GB\n", size, sizeof(double) * 
    (double)size * 1.e-9);
  printf("CPU total time is %f ms, bandwidth %f GB/s\n", CPUelapsedTime,
    sizeof(double)*(double)size / 
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


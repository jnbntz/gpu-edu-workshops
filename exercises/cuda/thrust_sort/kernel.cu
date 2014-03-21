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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

int main(void)
{
  // generate 32M random numbers on the host
  thrust::host_vector<int> h_vec( 32 << 20 );
  thrust::generate( h_vec.begin(), h_vec.end(), rand );

  // replicate input on another host vector
  thrust::host_vector<int> h_vec1 = h_vec;

  //transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  //create timers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );

  //sort data on the device
  thrust::sort( d_vec.begin(), d_vec.end() );

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float GPUelapsedTime;
  cudaEventElapsedTime( &GPUelapsedTime, start, stop );

  GPUelapsedTime /= 1000.0;

  printf("sort of %ld in %f seconds\n", 32<<20, GPUelapsedTime );
  printf("Sort of %f M / sec\n", (double)(32<<20) / (double)GPUelapsedTime *
   1e-6);

  //transfer data back to host
  thrust::copy( d_vec.begin(), d_vec.end(), h_vec.begin() );


  cudaEventRecord( start, 0 );

  //sort data on host
  thrust::sort(h_vec1.begin(), h_vec1.end() );

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float CPUelapsedTime;
  cudaEventElapsedTime( &CPUelapsedTime, start, stop );
  CPUelapsedTime /= 1000.0;

  printf("sort of %ld in %f seconds\n", 32<<20,CPUelapsedTime );
  printf("Sort of %f M / sec\n", (double)(32<<20) / (double)CPUelapsedTime *
   1e-6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("GPU is %5.2fX faster than CPU\n", CPUelapsedTime/GPUelapsedTime );

  if ( thrust::equal( h_vec1.begin(), h_vec1.end(), h_vec.begin() ) )
    printf("The arrays are equal\n");
  else
    printf("The arrays are different!\n");


  return 0;
} /* end main */


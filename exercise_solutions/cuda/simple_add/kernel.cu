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

__global__ void add(int *a, int *b, int *c)
{
  *c = *a + *b;
}

int main()
{
  int a, b, c;
  int *d_a, *d_b, *d_c;
  int size = sizeof( int );

/* allocate space for device copies of a, b, c */

  CUDA_CALL( cudaMalloc( (void **) &d_a, size ) );
  CUDA_CALL( cudaMalloc( (void **) &d_b, size ) );
  CUDA_CALL( cudaMalloc( (void **) &d_c, size ) );

/* setup initial values */

  a = 2;
  b = 7;
  c = -99;

/* copy inputs to device */

  CUDA_CALL( cudaMemcpy( d_a, &a, size, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpy( d_b, &b, size, cudaMemcpyHostToDevice ) );

/* launch the kernel on the GPU */

  add<<< 1, 1 >>>( d_a, d_b, d_c );
  CUDA_CHECK()
  CUDA_CALL( cudaDeviceSynchronize() );

/* copy result back to host */

  CUDA_CALL( cudaMemcpy( &c, d_c, size, cudaMemcpyDeviceToHost ) );

  printf("value of c after kernel is %d\n",c);
  if( c == ( a + b ) ) printf("PASS\n");
  else printf("FAIL\n");

/* clean up */

  CUDA_CALL( cudaFree( d_a ) );
  CUDA_CALL( cudaFree( d_b ) );
  CUDA_CALL( cudaFree( d_c ) );

  CUDA_CALL( cudaDeviceReset() );
	
  return 0;
} /* end main */

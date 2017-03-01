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

__global__ void add(int *a, int *b, int *c)
{
  *c = *a + *b;
}

int main()
{

  int a, b, c;
  int *d_a, *d_b, *d_c;
  int size = sizeof( int );

/* get GPU device number and name */
  
  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* allocate space for device copies of a, b, c */

  checkCUDA( cudaMalloc( (void **) &d_a, size ) );
  checkCUDA( cudaMalloc( (void **) &d_b, size ) );
  checkCUDA( cudaMalloc( (void **) &d_c, size ) );

/* setup initial values */

  a = 2;
  b = 7;
  c = -99;


/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_a, &a, size, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_b, &b, size, cudaMemcpyHostToDevice ) );

/* zero out the device memory for C */

  checkCUDA( cudaMemset( d_c, 0, size ) );

/* launch the kernel on the GPU */

  add<<< 1, 1 >>>( d_a, d_b, d_c );
  checkKERNEL()

/* copy result back to host */

  checkCUDA( cudaMemcpy( &c, d_c, size, cudaMemcpyDeviceToHost ) );

  printf("value of c after kernel is %d\n",c);
  if( c == ( a + b ) ) printf("PASS\n");
  else printf("FAIL\n");

/* clean up */

  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_b ) );
  checkCUDA( cudaFree( d_c ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

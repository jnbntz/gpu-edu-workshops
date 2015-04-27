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

__global__ void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main()
{
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof( int );

/* allocate space for device copies of a, b, c */

  checkCUDA( cudaMalloc( (void **) &d_a, size ) );
  checkCUDA( cudaMalloc( (void **) &d_b, size ) );
  checkCUDA( cudaMalloc( (void **) &d_c, size ) );

/* allocate space for host copies of a, b, c and setup input values */

  a = (int *)malloc( size );
  b = (int *)malloc( size );
  c = (int *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = i;
    c[i] = 0;
  }

/* copy inputs to device */

  checkCUDA( cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice ) );

/* launch the kernel on the GPU */
/* insert the launch parameters to launch properly using blocks and threads */
  add<<< FIXME, FIXME >>>( d_a, d_b, d_c );
  checkKERNEL()

/* copy result back to host */

  checkCUDA( cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost ) );

  for( int i = 0; i < N; i++ )
  {
    if( c[i] != a[i] + b[i] )
    {
      printf("c[%d] = %d\n",i,c[i] );
      printf("FAIL\n");
      goto end;
    } /* end if */
  }

  printf("PASS\n");
  end:

/* clean up */

  free(a);
  free(b);
  free(c);
  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_b ) );
  checkCUDA( cudaFree( d_c ) );

  checkCUDA( cudaDeviceReset() );
	
  return 0;
} /* end main */

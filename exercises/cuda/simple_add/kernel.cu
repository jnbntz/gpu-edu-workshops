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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define CUDA_ERROR() printf("cuda error is %s\n",cudaGetErrorString( cudaGetLastError() ));

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

	cudaMalloc( (void **) &d_a, size );
	/* enter code here to malloc d_b and d_c */
        FIXME

	/* setup initial values */

	a = 2;
	b = 7;
	c = -99;

	/* copy inputs to device */

	cudaMemcpy( d_a, &a, size, cudaMemcpyHostToDevice );
	/* enter code here to copy d_b to device */
        FIXME

	/* launch the kernel on the GPU */
	/* enter code here */
        FIXME

	/* copy result back to host */

	cudaMemcpy( &c, d_c, size, cudaMemcpyDeviceToHost );

	printf("value of c after kernel is %d\n",c);

	/* clean up */

	cudaFree( d_a );
	/* enter code here to cudaFree the d_b and d_c pointers */
	
	return 0;
} /* end main */

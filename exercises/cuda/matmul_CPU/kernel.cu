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

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* linear size of the matrices */

#define SIZE 1024


/* CPU matrix multiply function */

void host_dgemm( const int m, const int n, const int k, 
                 double const * const a, double const * const b, double *c )
{
	
/* 
 *  naive matrix multiplication loops go here.  triply nested for loop
 *  C = A * B where A and B are matrices
 *  C(i,j) = SUM( A(i,k) * B(k,j), over the index "k", where 0 <= k < (SIZE-1) )   
 */     
        

/* insert code here */

for( int j = 0; j < n; j++ )
{
  for( int i = 0; i < m; i++ )
  {
    for( int koff = 0; koff < k; koff++ )
    {
      c[INDX( FIXME )] += a[INDX( FIXME )] * b[INDX( FIXME )];
    } /* end for koff */
  } /* end for i */
} /* end for j */

} /* end host_dgemm */

int main( int argc, char *argv[] )
{

  int size = SIZE;

  fprintf(stdout, "Matrix size is %d\n",size);

/* declare host pointers */

  double *h_a, *h_b, *h_cdef;
 
  size_t numbytes = size * size * sizeof( double );

/* allocate host pointers */

  h_a = (double *) malloc( numbytes );
  if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_b = (double *) malloc( numbytes );
  if( h_b == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_cdef = (double *) malloc( numbytes );
  if( h_cdef == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

/* set C to zero */

  memset( h_cdef, 0, numbytes );

  fprintf( stdout, "Total memory required is %lf MB\n", 
       3.0 * (double) numbytes / 1000000.0 );

/* initialize A and B on the host */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  }

/* start timers */

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );

/* call host dgemm */

  host_dgemm( size, size, size, h_a, h_b, h_cdef );

/* stop the timers */

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop );

/* print the results */

  fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", 
    2.0 * (double) size * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* cleanup */

  free( h_a );
  free( h_b );
  free( h_cdef );

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;
}

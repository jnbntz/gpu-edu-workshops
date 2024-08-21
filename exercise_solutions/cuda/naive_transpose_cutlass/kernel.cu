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

#include <iostream>
#include "../debug.h"
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

/* definitions of threadblock size in X and Y directions */

const int threads_per_block_x = 32;
const int threads_per_block_y = 32;

/* definition of matrix linear dimension */

const int matSize = 4096;

/* macro to index a 1D memory array with 2D indices in column-major order */
inline __host__ __device__ long int indx( const long int row, const long int col, const long int ld ) 
{
  return (col * ld) + row;
} /* end indx */

/* CUDA kernel for naive matrix transpose */

__global__ void naive_cuda_transpose( const int m, 
                                      const double * const a, 
                                            double * const c )
{
  const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
  const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

  if( myRow < m && myCol < m )
  {
    c[indx( myCol, myRow, m )] = a[indx( myRow, myCol, m )];
  } /* end if */
  return;

} /* end naive_cuda_transpose */

void host_transpose( const int m, const thrust::host_vector<double> &a, 
								        thrust::host_vector<double> &c )
{
	
/* 
 *  naive matrix transpose goes here.
 */
 
  for( int j = 0; j < m; j++ )
  {
    for( int i = 0; i < m; i++ )
      {
        c[indx(i,j,m)] = a[indx(j,i,m)];
      } /* end for i */
  } /* end for j */

} /* end host transpose */

int main( int argc, char *argv[] )
{

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  cout << "Using GPU " << dev << ": " << deviceProp.name << endl;

  int size = matSize;

  cout << "Matrix size is " << size << endl;

/* declaring pointers for array */

  size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* allocating and set host memory */

  thrust::host_vector<double> h_a( size * size );
  thrust::generate( h_a.begin(), h_a.end(), rand );

  thrust::host_vector<double> h_c( size * size, 0.0 );

/* allocating and set device memory and copy a from host to device */

  thrust::device_vector<double> d_a = h_a;
  thrust::device_vector<double> d_c( size * size, 0.0 );

  cout << "size of vector is " << d_a.size() << endl;
  cout << "Total memory required per matrix is " << (double) numbytes / 1000000.0 << endl;

/* create and start timer */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call naive cpu transpose function */

  host_transpose( size, h_a, h_c );

/* stop CPU timer */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print CPU timing information */

  cout << "Total time CPU is " << elapsedTime / 1000.0f << " sec" << endl;
  cout << "Performance is " << 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 << endl ; 

/* setup threadblock size and grid sizes */

  dim3 threads( threads_per_block_x, threads_per_block_y, 1 );
  dim3 blocks( ( size / threads_per_block_x ) + 1, 
               ( size / threads_per_block_y ) + 1, 1 );

/* case the thrust device pointers to raw pointers so we can pass to kernel */

  double *raw_d_a = thrust::raw_pointer_cast(d_a.data());
  double *raw_d_c = thrust::raw_pointer_cast(d_c.data());

/* start timers */
  checkCUDA( cudaEventRecord( start, 0 ) );
  
/* call naive GPU transpose kernel */

  naive_cuda_transpose<<< blocks, threads >>>( size, raw_d_a, raw_d_c );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU timing information */

  cout << "Total time GPU is " << elapsedTime / 1000.0f << " sec" << endl;
  cout << "Performance is " << 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 << endl ; 

/* set the host array back to zero */

  thrust::fill( h_a.begin(), h_a.end(), 0.0 );

/* copy data back from device to host */

  h_a = d_c;

/* compare GPU to CPU for correctness */

  if ( thrust::equal( h_a.begin(), h_a.end(), h_c.begin() ) )
    printf("The arrays are equal\n");
  else
    printf("The arrays are different!\n");

  int success = 1;

  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[indx(i,j,size)] != h_a[indx(i,j,size)] ) 
      {
		cerr << "Error in element " << i << "," << j << endl;
		cerr << "Host " << h_c[indx(i,j,size)] << ", device " << h_a[indx(i,j,size)] << endl;
        success = 0;
        break;
      } /* end fi */
    } /* end for i */
  } /* end for j */

  if( success == 1 ) cout << "PASS" << endl;
  else               cout << "FAIL" << endl;

//  checkCUDA( cudaDeviceReset() );

  return 0;
} /* end main */

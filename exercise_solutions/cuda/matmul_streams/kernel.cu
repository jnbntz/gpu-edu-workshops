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

#ifdef DEBUG

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

#define CUBLAS_CALL(F)  if( (F) != CUBLAS_STATUS_SUCCESS ) \
  {printf("Error %d at %s:%d\n", F , \
   __FILE__,__LINE__); exit(-1);}

#else

#define CUDA_CALL(F) (F)
#define CUDA_CHECK() ()
#define CUBLAS_CALL(F) (F)

#endif

#include <stdio.h>
#include "cublas_v2.h"

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

#define TILESIZE 2048
#define SIZE ( TILESIZE * 4 )
#define NUM_STREAMS 3

__global__ void printMat( const double *A, int size )
{
  if( threadIdx.x == 0 && blockIdx.x == 0 )
    for( int i = 0; i < size; i++ )
      printf("A[%d] = %f\n",i,A[i]);
  return;
} /* end printMat */

void printMatHost( const double *A, int size )
{
  for( int i = 0; i < size; i++ )
    printf("A[%d] = %f\n",i,A[i]);
  return;
} /* end printMatHost */


int main( int argc, char *argv[] )
{

    const int size = SIZE;
    const int nstreams = NUM_STREAMS;
    const int tileSize = TILESIZE;

    fprintf(stdout, "\nMatrix size is %d by %d\n",size, size);
    fprintf(stdout, "Tile size is %d by %d\n",tileSize, tileSize);
    fprintf(stdout, "Number of streams is %d\n\n",nstreams);

    double *h_a, *h_b, *h_c, *h_cdef;
    double *p_a, *p_b, *p_c;
    double **d_a, **d_b, **d_c;

    size_t numbytes = size * size * sizeof( double );
    size_t tileBytes = tileSize * tileSize * sizeof( double );

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

    h_c = (double *) malloc( numbytes );
    if( h_c == NULL )
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

    memset( h_c, 0, numbytes );
    memset( h_cdef, 0, numbytes );

    fprintf(stdout, "---Standard CUBLAS with synchronous memory transfer---\n");

    fprintf( stdout, "Total CPU memory required is %lf MB\n",
       3.0 * (double) numbytes / 1000000.0 );

    fprintf( stdout, "Total GPU memory required is %lf MB\n",
       3.0 * (double) numbytes / 1000000.0 );

    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
      h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
//      h_a[i] = ( i / size ) == (i % size) ? 1.0 : 0.0;
 //     h_b[i] = (double) i;
    }

    memset( h_c, 0, numbytes );
    memset( h_cdef, 0, numbytes );

    d_a = (double **) malloc( sizeof( double *) * nstreams );
    d_b = (double **) malloc( sizeof( double *) * nstreams );
    d_c = (double **) malloc( sizeof( double *) * nstreams );

    CUDA_CALL( cudaMalloc( (void **)&d_a[0], numbytes ) );
    CUDA_CALL( cudaMalloc( (void **)&d_b[0], numbytes ) );
    CUDA_CALL( cudaMalloc( (void **)&d_c[0], numbytes ) );


/* CUBLAS test for sanity */

    cudaEvent_t start, stop;
    CUDA_CALL( cudaEventCreate( &start ) );
    CUDA_CALL( cudaEventCreate( &stop ) );
    float elapsedTime;

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    double alpha = 1.0;
    double beta  = 1.0;

    CUDA_CALL( cudaEventRecord( start, 0 ) );

    CUDA_CALL( cudaMemcpy( d_a[0], h_a, numbytes, cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_b[0], h_b, numbytes, cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_c[0], h_c, numbytes, cudaMemcpyHostToDevice ) );

    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha,
                 d_a[0], size,
                 d_b[0], size,
                 &beta,
                 d_c[0], size );

    CUDA_CALL( cudaMemcpy( h_cdef, d_c[0], numbytes, cudaMemcpyDeviceToHost ) );

    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n",
            elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n",
      2.0 * (double) size * (double) size * (double) size /
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

    CUDA_CALL( cudaFree( d_a[0] ) );
    CUDA_CALL( cudaFree( d_b[0] ) );
    CUDA_CALL( cudaFree( d_c[0] ) );

/* end cublas test for sanity */

/* setup for cublas wtih streams */
    fprintf(stdout, "\n---Streams CUBLAS with asynchronous memory transfer---\n");
    fprintf( stdout, "Total CPU memory required is %lf MB\n",
       3.0 * (double) numbytes / 1000000.0 * 2.0 );

    fprintf( stdout, "Total GPU memory required is %lf MB\n",
       3.0 * (double) tileBytes / 1000000.0 );

/* cudamalloc device memory for each stream */

    for( int j = 0; j < nstreams; j++ )
    {
      CUDA_CALL( cudaMalloc( &d_a[j], tileBytes ) );
      CUDA_CALL( cudaMalloc( &d_b[j], tileBytes ) );
      CUDA_CALL( cudaMalloc( &d_c[j], tileBytes ) );
    } /* end for */

/* cudamallochost for pinned host memory */

    CUDA_CALL( cudaMallocHost( &p_a, numbytes ) );
    CUDA_CALL( cudaMallocHost( &p_b, numbytes ) );
    CUDA_CALL( cudaMallocHost( &p_c, numbytes ) );

/* copy matrix into pinned memory in tiles */

    int pinnedOffset = 0;

    for( int colTile = 0; colTile < size; colTile += tileSize )
    {
      for( int rowTile = 0; rowTile < size; rowTile += tileSize )
      {
        for( int col = 0; col < tileSize; col++ )
        {
          for( int row = 0; row < tileSize; row++ )
          {
            int pagedOffset = INDX( rowTile + row, colTile + col, size );

            p_a[ pinnedOffset ] = h_a[ pagedOffset ];
            p_b[ pinnedOffset ] = h_b[ pagedOffset ];
            p_c[ pinnedOffset ] = h_c[ pagedOffset ];

            pinnedOffset++;
          } /* end for */
        } /* end for */
      } /* end for */
    } /* end for */

/* create the streams */

    int *rowTile, *colTile;
    rowTile = (int *) malloc( nstreams * sizeof(int) );
    colTile = (int *) malloc( nstreams * sizeof(int) );

    cudaStream_t stream[nstreams];
    for( int i = 0; i < nstreams; i++ )
    {
/* create the streams here */
        CUDA_CALL( cudaStreamCreate( &stream[i] ) );
    } /* end for */

    cublasDestroy( handle );
    stat = cublasCreate( &handle );

    int linearTiles = ( size / tileSize );
    int totalTiles = linearTiles * linearTiles;

    int currentTile = 0;

/* starting the timer */

    CUDA_CALL( cudaEventRecord( start, 0 ) );

/* while loop over all the tiles */

    while( currentTile < totalTiles )
    {

      int localStreams = 0;

/* assign a tile to each stream */

      for( int i = 0; i < nstreams; i++ )
      {
        rowTile[i] = currentTile % linearTiles;
        colTile[i] = currentTile / linearTiles;

/*   using localStreams in case we run out of tiles and still have streams
     then we just keep some streams idle  */

        if( currentTile < totalTiles ) localStreams++;

        currentTile++;
      } /* end for */

/* copy tile of C to device, each stream does a different tile of C */

      for( int i = 0; i < localStreams; i++ )
      {
        int cOffset = INDX( rowTile[i], colTile[i], linearTiles ) *
                            tileSize * tileSize;
        CUDA_CALL( cudaMemcpyAsync( d_c[i], &p_c[cOffset], tileBytes,
                     cudaMemcpyHostToDevice, stream[i] ) );

      } /* end for */


/* inner K loop of matmult */

      for( int k = 0; k < linearTiles; k++ )
      {

/* stream A into device */

        for( int i = 0; i < localStreams; i++ )
        {

        int aOffset = INDX( rowTile[i], k, linearTiles ) *
                            tileSize * tileSize;
        CUDA_CALL( cudaMemcpyAsync( d_a[i], &p_a[aOffset], tileBytes,
                     cudaMemcpyHostToDevice, stream[i] ) );
        } /* end for */

/* stream B into device */

        for( int i = 0; i < localStreams; i++ )
        {
        int bOffset = INDX( k, colTile[i], linearTiles ) *
                            tileSize * tileSize;
        CUDA_CALL( cudaMemcpyAsync( d_b[i], &p_b[bOffset], tileBytes,
                     cudaMemcpyHostToDevice, stream[i] ) );

        } /* end for */

/* do the dgemm */

        for( int i = 0; i < localStreams; i++ )
        {
/* set the stream for the cublas call */

          CUBLAS_CALL( cublasSetStream( handle, stream[i] ) );

/* call the cublas dgemm function */

          CUBLAS_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 tileSize, tileSize, tileSize,
                 &alpha,
                 d_a[i], tileSize,
                 d_b[i], tileSize,
                 &beta,
                 d_c[i], tileSize ) );

        } /* end for */

      }/* end for k */

/* copy tile of C back to host */

      for( int i = 0; i < localStreams; i++ )
      {
        int cOffset = INDX( rowTile[i], colTile[i], linearTiles ) *
                            tileSize * tileSize;
        CUDA_CALL( cudaMemcpyAsync( &p_c[cOffset], d_c[i], tileBytes,
                        cudaMemcpyDeviceToHost, stream[i] ) );
      } /* end for */


    } /* end while */

    CUDA_CALL( cudaDeviceSynchronize() );
    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time GPU Streams with CUBLAS is %f sec\n",
            elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n",
      2.0 * (double) size * (double) size * (double) size /
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

    free( rowTile );
    free( colTile );

//   printMatHost( p_c, size * size );

    pinnedOffset = 0;

    for( int colTile = 0; colTile < size; colTile += tileSize )
    {
      for( int rowTile = 0; rowTile < size; rowTile += tileSize )
      {
        for( int col = 0; col < tileSize; col++ )
        {
          for( int row = 0; row < tileSize; row++ )
          {
            int pagedOffset = INDX( rowTile + row, colTile + col, size );
            h_c[ pagedOffset ] = p_c[ pinnedOffset ];

            pinnedOffset++;
          } /* end for */
        } /* end for */
      } /* end for */
    } /* end for */


    double temp = 0.0;
    for( int i = 0; i < size * size; i++ )
    {
      temp += ( h_c[i] - h_cdef[i] ) * ( h_c[i] - h_cdef[i] );
//      printf("i %d host %f device %f\n",i, h_cdef[i], h_c[i] );
    } /* end for */
    printf("error is %f\n",temp);
    if( temp > 10 ) printf("FAIL\n");
    else printf("PASS\n");

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_cdef );

    for( int i = 0; i < nstreams; i++ )
    {
/* destroy the streams here */
        CUDA_CALL( cudaStreamDestroy( stream[i] ) );
    } /* end for */

    CUDA_CALL( cudaFreeHost( p_a ) );
    CUDA_CALL( cudaFreeHost( p_b ) );
    CUDA_CALL( cudaFreeHost( p_c ) );

    CUDA_CALL( cudaDeviceReset() );
    return 0;
}


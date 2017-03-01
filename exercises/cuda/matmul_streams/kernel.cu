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
#include "cublas_v2.h"
#include "../debug.h"

typedef float floatType_t;

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

#define TILESIZE 2048
#define SIZE ( TILESIZE * 4 )
#define NUM_STREAMS 3

__global__ void printMat( const floatType_t *A, int size )
{
  if( threadIdx.x == 0 && blockIdx.x == 0 )
    for( int i = 0; i < size; i++ )
      printf("A[%d] = %f\n",i,A[i]);
  return;
} /* end printMat */

void printMatHost( const floatType_t *A, int size )
{
  for( int i = 0; i < size; i++ )
    printf("A[%d] = %f\n",i,A[i]);
  return;
} /* end printMatHost */


int main( int argc, char *argv[] )
{
/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

    const int size = SIZE;
    const int nstreams = NUM_STREAMS;
    const int tileSize = TILESIZE;

    fprintf(stdout, "\nMatrix size is %d by %d\n",size, size);
    fprintf(stdout, "Tile size is %d by %d\n",tileSize, tileSize);
    fprintf(stdout, "Number of streams is %d\n\n",nstreams);

    floatType_t *h_a, *h_b, *h_c, *h_cdef;
    floatType_t *p_a, *p_b, *p_c;
    floatType_t **d_a, **d_b, **d_c;

    size_t numbytes = size * size * sizeof( floatType_t );
    size_t tileBytes = tileSize * tileSize * sizeof( floatType_t );

    h_a = (floatType_t *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_b = (floatType_t *) malloc( numbytes );
    if( h_b == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_c = (floatType_t *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_cdef = (floatType_t *) malloc( numbytes );
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

    d_a = (floatType_t **) malloc( sizeof( floatType_t *) * nstreams );
    d_b = (floatType_t **) malloc( sizeof( floatType_t *) * nstreams );
    d_c = (floatType_t **) malloc( sizeof( floatType_t *) * nstreams );

    checkCUDA( cudaMalloc( (void **)&d_a[0], numbytes ) );
    checkCUDA( cudaMalloc( (void **)&d_b[0], numbytes ) );
    checkCUDA( cudaMalloc( (void **)&d_c[0], numbytes ) );


/* CUBLAS test for sanity */

    cudaEvent_t start, stop;
    checkCUDA( cudaEventCreate( &start ) );
    checkCUDA( cudaEventCreate( &stop ) );
    float elapsedTime;

    cublasHandle_t handle;
    checkCUBLAS( cublasCreate( &handle ) );

    floatType_t alpha = 1.0;
    floatType_t beta  = 1.0;

    checkCUDA( cudaEventRecord( start, 0 ) );

    checkCUDA( cudaMemcpy( d_a[0], h_a, numbytes, cudaMemcpyHostToDevice ) );
    checkCUDA( cudaMemcpy( d_b[0], h_b, numbytes, cudaMemcpyHostToDevice ) );
    checkCUDA( cudaMemcpy( d_c[0], h_c, numbytes, cudaMemcpyHostToDevice ) );

    if( sizeof(floatType_t) == 4 )
    {
    checkCUBLAS(
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 (float *)&alpha,
                 (float *)d_a[0], size,
                 (float *)d_b[0], size,
                 (float *)&beta,
                 (float *)d_c[0], size )
               );
    }
    else
    {
    checkCUBLAS( 
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 (double *)&alpha,
                 (double *)d_a[0], size,
                 (double *)d_b[0], size,
                 (double *)&beta,
                 (double *)d_c[0], size )
               );
    }

    checkCUDA( cudaMemcpy( h_cdef, d_c[0], numbytes, cudaMemcpyDeviceToHost ) );

    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n",
            elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n",
      2.0 * (double) size * (double) size * (double) size /
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

    checkCUDA( cudaFree( d_a[0] ) );
    checkCUDA( cudaFree( d_b[0] ) );
    checkCUDA( cudaFree( d_c[0] ) );

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
      checkCUDA( cudaMalloc( &d_a[j], tileBytes ) );
      checkCUDA( cudaMalloc( &d_b[j], tileBytes ) );
      checkCUDA( cudaMalloc( &d_c[j], tileBytes ) );
    } /* end for */

/* cudamallochost for pinned host memory */

    checkCUDA( cudaMallocHost( &p_a, numbytes ) );
    checkCUDA( cudaMallocHost( &p_b, numbytes ) );
    checkCUDA( cudaMallocHost( &p_c, numbytes ) );

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
      FIXME
    } /* end for */

    checkCUBLAS( cublasDestroy( handle ) );
    checkCUBLAS( cublasCreate( &handle ) );

    int linearTiles = ( size / tileSize );
    int totalTiles = linearTiles * linearTiles;

    int currentTile = 0;

/* starting the timer */

    checkCUDA( cudaEventRecord( start, 0 ) );

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
/* use cudaMemcpyAsync to move a tile of C to the device
   HINT: cOffset points to the starting location of the tile, in host mem */
        FIXME
      } /* end for */


/* inner K loop of matmult */

      for( int k = 0; k < linearTiles; k++ )
      {

/* stream A into device */

        for( int i = 0; i < localStreams; i++ )
        {

        int aOffset = INDX( rowTile[i], k, linearTiles ) *
                            tileSize * tileSize;
/* use cudaMemcpyAsync to move a tile of A to the device
   HINT: aOffset points to the starting location of the tile, in host mem */
        FIXME
        } /* end for */

/* stream B into device */

        for( int i = 0; i < localStreams; i++ )
        {
        int bOffset = INDX( k, colTile[i], linearTiles ) *
                            tileSize * tileSize;
/* use cudaMemcpyAsync to move a tile of A to the device
   HINT: bOffset points to the starting location of the tile, in host mem */
        FIXME

        } /* end for */

/* do the gemm */

        for( int i = 0; i < localStreams; i++ )
        {
/* set the stream for the cublas call */

          FIXME

/* call the cublas gemm function */

          FIXME

        } /* end for */

      }/* end for k */

/* copy tile of C back to host */

      for( int i = 0; i < localStreams; i++ )
      {
        int cOffset = INDX( rowTile[i], colTile[i], linearTiles ) *
                            tileSize * tileSize;
/* use cudaMemcpyAsync to move a tile of C to the host
   HINT: cOffset points to the starting location of the tile, in host mem */
        FIXME
      } /* end for */


    } /* end while */

    checkCUDA( cudaDeviceSynchronize() );
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

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
        temp = max( temp, abs( (double)h_c[i] - (double)h_cdef[i] )/
                      abs((double)h_cdef[i]) );
    } /* end for */
    printf("Maximum error is %e percent \n",temp*100.0);
    if( temp > 0.001 ) printf("FAIL\n");
    else printf("PASS\n");

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_cdef );

    for( int i = 0; i < nstreams; i++ )
    {
/* destroy the streams here */
      FIXME
    } /* end for */

    checkCUDA( cudaFreeHost( p_a ) );
    checkCUDA( cudaFreeHost( p_b ) );
    checkCUDA( cudaFreeHost( p_c ) );

    checkCUDA( cudaDeviceReset() );
    return 0;
}


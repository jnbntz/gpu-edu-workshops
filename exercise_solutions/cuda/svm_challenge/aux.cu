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
#include <stdlib.h>
#include <math.h>
#include "headers.h"
#include "cublas_v2.h"
#include "kernels.h"

#define INDX(row,col,ld) (((col) * (ld)) + (row))

void svmTrain( floatType_t const *d_X, 
               floatType_t const *d_y, 
               floatType_t const C,
               int const numFeatures, int const numTrainingExamples,
               floatType_t const tol, int const maxPasses, 
               floatType_t *d_W )
{

/* declare pointers for arrays */
  floatType_t *d_K, *d_alphas, *d_f;

/* declare variables */
  floatType_t bHigh, bLow;
  int ILow, IHigh;

/* device variables */

  floatType_t *d_bLow, *d_bHigh;
  int *d_ILow, *d_IHigh;

/* cuBLAS data */
  cublasStatus_t stat;
  cublasHandle_t handle;
  floatType_t const alpha=1.0; 
  floatType_t const beta =0.0;

/* malloc alphas */

  CUDA_CALL( cudaMalloc( (void**) &d_alphas, 
               sizeof(floatType_t) * numTrainingExamples ) );

/* zero alphas */

  CUDA_CALL( cudaMemset( d_alphas, 0, 
               sizeof(floatType_t)*numTrainingExamples ) );

/* malloc f */

  CUDA_CALL( cudaMalloc( (void**) &d_f,
               sizeof(floatType_t) * numTrainingExamples ) );

  int threads_per_block = 256;
  k_initF<<<TRAINING_SET_SIZE/threads_per_block+1,threads_per_block>>>
                      ( d_f, d_y, numTrainingExamples );

/* malloc K, the kernel matrix */

  CUDA_CALL( cudaMalloc( (void**) &d_K,
           sizeof(floatType_t) * numTrainingExamples * numTrainingExamples ) );
  CUDA_CALL( cudaMemset( d_K, 0, 
           sizeof(floatType_t)*numTrainingExamples*numTrainingExamples ));

/* compute the Kernel on every pair of examples.
   K = X * X'
   Wouldn't do this in real life especially if X was really large.  
   For large K we'd just calculate the rows needed on the fly in the
   large loop
*/

  stat = cublasCreate( &handle );
  if( stat != CUBLAS_STATUS_SUCCESS ) printf("error creating cublas handle\n");

  if( sizeof( floatType_t ) == sizeof( float ) )
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
               numTrainingExamples, numTrainingExamples, numFeatures,
               (float *)&alpha, (float *)d_X, numTrainingExamples,
               (float *)d_X, numTrainingExamples, (float *)&beta,
               (float *)d_K, numTrainingExamples );
   
  }
  else
  {
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
               numTrainingExamples, numTrainingExamples, numFeatures,
               (double *)&alpha, (double *)d_X, numTrainingExamples,
               (double *)d_X, numTrainingExamples, (double *)&beta,
               (double *)d_K, numTrainingExamples );
  }

  CUDA_CALL( cudaMalloc( (void**)&d_bLow, sizeof(floatType_t) ) );
  CUDA_CALL( cudaMalloc( (void**)&d_bHigh, sizeof(floatType_t) ) );
  CUDA_CALL( cudaMalloc( (void**)&d_ILow, sizeof(int) ) );
  CUDA_CALL( cudaMalloc( (void**)&d_IHigh, sizeof(int) ) );

  while( true )
  {

/* calculate the bLow and bHigh.  Must be called with only one 
   threadblock because it does a reduction.  Now what we'd do in practice
   but since the data size is small we can get away with it.
*/

    k_calculateBI<<<1,128>>>( d_f, d_alphas, d_y, numTrainingExamples,
                            d_bLow, d_bHigh, d_ILow, d_IHigh, C );
    CUDA_CHECK()
    CUDA_CALL( cudaDeviceSynchronize() );

    CUDA_CALL( cudaMemcpy( &bLow, d_bLow, sizeof(floatType_t),
                           cudaMemcpyDeviceToHost ) );
    CUDA_CALL( cudaMemcpy( &bHigh, d_bHigh, sizeof(floatType_t),
                           cudaMemcpyDeviceToHost ) );
    CUDA_CALL( cudaMemcpy( &ILow, d_ILow, sizeof(int),
                           cudaMemcpyDeviceToHost ) );
    CUDA_CALL( cudaMemcpy( &IHigh, d_IHigh, sizeof(int),
                           cudaMemcpyDeviceToHost ) );

/* exit loop once we are below tolerance level */     
    if( bLow <= ( bHigh + ((floatType_t) 2.0 * tol) ) ) 
      break; 

/* update f array */

    k_updateF<<<TRAINING_SET_SIZE/threads_per_block + 1,threads_per_block>>>
                  ( d_f, d_alphas,
                           IHigh,
                           ILow,
                           d_K, numTrainingExamples, d_y, C, 
                           bLow, bHigh );
    CUDA_CHECK()
    CUDA_CALL( cudaDeviceSynchronize() );

  } /* end while */

  k_scaleAlpha<<<TRAINING_SET_SIZE/threads_per_block + 1,threads_per_block>>>
                   ( d_alphas, d_y, numTrainingExamples );
  CUDA_CHECK()
  CUDA_CALL( cudaDeviceSynchronize() );

/* calculate W from alphas */

  if( sizeof( floatType_t ) == sizeof( float ) )
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               1, numFeatures, numTrainingExamples,
               (float *)&alpha, (float *)d_alphas, 1,
               (float *)d_X, numTrainingExamples, (float *)&beta,
               (float *)d_W, 1 );
  }
  else
  {
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               1, numFeatures, numTrainingExamples,
               (double *)&alpha, (double *)d_alphas, 1,
               (double *)d_X, numTrainingExamples, (double *)&beta,
               (double *)d_W, 1 );
  }
  
  CUDA_CALL( cudaFree( d_alphas ) );
  CUDA_CALL( cudaFree( d_f ) );
  CUDA_CALL( cudaFree( d_K ) );
  CUDA_CALL( cudaFree( d_ILow ) );
  CUDA_CALL( cudaFree( d_IHigh ) );
  CUDA_CALL( cudaFree( d_bLow ) );
  CUDA_CALL( cudaFree( d_bHigh ) );

} /* end svmTrain */

void svmPredict( floatType_t const *X, 
                 floatType_t const *W, 
                 int const numExamples, int const numFeatures,
                 int *pred )
{
  floatType_t *p;

  p = (floatType_t *) malloc( sizeof(floatType_t) * numExamples );
  if( p == NULL ) fprintf(stderr,"error in malloc p in svmTrain\n");

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemv( CblasColMajor, CblasNoTrans,
               numExamples, numFeatures,
               1.0, (float *)X, numExamples,
               (float *)W, 1, 0.0,
               (float *)p, 1 );
  }
  else
  {
    cblas_dgemv( CblasColMajor, CblasNoTrans,
               numExamples, numFeatures,
               1.0, (double *)X, numExamples,
               (double *)W, 1, 0.0,
               (double *)p, 1 );
  }

  for( int i = 0; i < numExamples; i++ )
    pred[i] = ( p[i] >= 0.0 ) ? 1 : 0;
 
  free(p);
  return;
} /* end svmTrain */

void readMatrixFromFile( char *fileName, 
                         int *matrix, 
                         int const rows, 
                         int const cols )
{
  FILE *ifp;

  ifp = fopen( fileName, "r" );

  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening file %s\n", fileName);
    exit(911);
  } /* end if */

  for( int row = 0; row < rows; row++ )
  {
    for( int col = 0; col < cols; col++ )
    {
      if( !fscanf( ifp, "%d", 
          &matrix[ INDX( row, col, rows ) ] ) )
      {
        fprintf(stderr,"error reading training matrix file \n");
        exit(911);
      } /* end if */
    } /* end for col */
  } /* end for row */

  fclose(ifp);
  return;
} /* end readMatrixFromFile */

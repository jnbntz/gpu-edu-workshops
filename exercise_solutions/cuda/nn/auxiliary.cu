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
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include "headers.h"

cublasHandle_t cublasHandle;

/* kernel to update delta2 parameter */

__global__ void k_updateDelta2( floatType_t       *delta2,
                                floatType_t const *z2,
                                int         const Xexamples,
                                int         const size )
{

/* setup global threadID in X and Y directions */

  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  if( tidy < Xexamples && tidx < size )
  {

/* calculate the offset properly */

    delta2[INDX(tidx,tidy,size)] *= z2[INDX(tidy,tidx,Xexamples)];

  } /* end if */
} /* end k_updateDelta2 */

/* kernel for calculating the sigmoid of an array */

__global__ void k_sigmoid_f( floatType_t  *array,
                                     int    const size )
{

/* setup global threadID in X direction */

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if( tid < size )
  {

/* use the sigmoid_f function to complete this loop */

    array[tid] = sigmoid_f( array[tid] );

  } /* end if */
} /* end sigmoidGradient */

/* kernel for calculating the gradient of the sigmoid function */

__global__ void k_sigmoidGradient_f( floatType_t  *array,
                                     int    const size )
{

/* setup global threadID in X direction */

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if( tid < size )
  {

/* use the sigmoidGradient_f function to complete this loop */

    array[tid] = sigmoidGradient_f( array[tid] );

  } /* end if */
} /* end sigmoidGradient */

/* kernel to set the delta3 vector from Y properly 
   delta3 is just the different between the calculated value
   a3 and the true value Y3
*/

__global__ void  setDelta3Vec( floatType_t       *delta3, 
                               floatType_t const *Y, 
                               floatType_t const *a3,
                               int         const  Xexamples )
{

/* setup global threadID in X direction */

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if( tid < Xexamples )
  {
    delta3[INDX((int)Y[tid],tid,11)] = (floatType_t) 1.0;
    for( int j = 0; j < 10; j++ )
    {
      delta3[INDX(j+1,tid,11)] = a3[INDX(tid,j,Xexamples)]
                               - delta3[INDX(j+1,tid,11)];
    } /* end for j */
  }
  return;
} /* end setDelta3Vec */

/* init the array to 1.0. used to set the bias term */

__global__ void initOne( int size, floatType_t *array )
{

/* setup global threadID in X direction */

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if( tid < size )
    array[tid] = (floatType_t) 1.0;
  return;
} /* end initOne */

/* debugging kernel for printing values */

__global__ void printKernel( int rows, int cols, floatType_t *array )
{
  for( int j = 0; j < cols; j++ )
  {
    for( int i = 0; i < rows; i++ )
    {
      printf("row %d col %d value %e\n",i,j,array[INDX(i,j,rows)] );
    } /* end for */
  } /* end for */
} /* end print Kernel */

/* debugging for printing on host */

void printHost( int rows, int cols, floatType_t *array )
{
  for( int j = 0; j < cols; j++ )
  {
    for( int i = 0; i < rows; i++ )
    {
      printf("row %d col %d value %e\n",i,j,array[INDX(i,j,rows)] );
    } /* end for */
  } /* end for */
} /* end print Kernel */

/* main function to train the network */

void trainNetwork( floatType_t       *X, 
                   int         const Xexamples, 
                   int         const Xfeatures,
                   floatType_t       *theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t       *theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y, 
                   float       const learningRate,
                   int         const iterations,
                   int         const batchSize )
{
  floatType_t lambda = learningRate;
  floatType_t cost;

  checkCUBLAS( cublasCreate( &cublasHandle ) );

/* allocate large GPU space for temporary arrays */

  floatType_t *d_tempMatrix;
  checkCUDA( cudaMalloc( &d_tempMatrix, sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + //z2 
                                 Xexamples * (theta1Rows+1) + //a2
                                 Xexamples * (theta2Rows+1) + //a3
                                 Xexamples * (theta1Rows+1) + //delta2
                                 Xexamples * 11 ) ) );            //delta3

/* set the bias term to 1, sort of like the y-intercept */

  for( int i = 0; i < Xexamples; i++ ) 
  {
    X[INDX(0,i,Xfeatures)] = (floatType_t) 1.0;
  } /* end for */

/* malloc all arrays I need on the device and copy data from the host */

  floatType_t *d_X;
  checkCUDA( cudaMalloc( &d_X, sizeof(floatType_t)*Xexamples*Xfeatures));
  checkCUDA( cudaMemcpy( d_X, X, 
                         sizeof(floatType_t)*Xexamples*Xfeatures,
                         cudaMemcpyHostToDevice ) );

  floatType_t *d_Y;
  checkCUDA( cudaMalloc( &d_Y, sizeof(floatType_t)*Xexamples) );
  checkCUDA( cudaMemcpy( d_Y, Y, 
                         sizeof(floatType_t)*Xexamples,
                         cudaMemcpyHostToDevice ) );

/* theta1 and theta2 are the weights in the 1st and second layers */

  floatType_t *d_theta1;
  checkCUDA( cudaMalloc( &d_theta1, 
          sizeof(floatType_t) * theta1Rows * theta1Cols ) );

  checkCUDA( cudaMemcpy( d_theta1, theta1,
                         sizeof(floatType_t)*theta1Rows*theta1Cols,
                         cudaMemcpyHostToDevice ) );

  floatType_t *d_theta2;
  checkCUDA( cudaMalloc( &d_theta2, 
          sizeof(floatType_t) * theta2Rows * theta2Cols ) );

  checkCUDA( cudaMemcpy( d_theta2, theta2,
                         sizeof(floatType_t)*theta2Rows*theta2Cols,
                         cudaMemcpyHostToDevice ) );

/* theta1Grad and theta2Grad are the gradients for theta1 and theta2 */

  floatType_t *d_theta1Grad, *d_theta2Grad;
  checkCUDA( cudaMalloc( &d_theta1Grad, 
                         sizeof(floatType_t)*theta1Rows*theta1Cols ) );

  checkCUDA( cudaMalloc( &d_theta2Grad,
                         sizeof(floatType_t)*theta2Rows*theta2Cols ) );

/* stochastic gradient descent 
   in our case stochastic because the data is already scrambled */

  int iter = 0;

/* big while loop over the number of iterations to train */

  while(iter < iterations )
  {

/* for loop over the batch size */

    for( int j = 0; j < Xexamples; j+=batchSize )
    {

      int tempBatchSize = min( batchSize, Xexamples - j );

/* bulk of computation here */

      calcGradient( &d_X[INDX(0,j,Xfeatures)], tempBatchSize, Xfeatures,
                    d_theta1, theta1Rows, theta1Cols, 
                    d_theta2, theta2Rows, theta2Cols,
                    &d_Y[j],
                    &cost, d_theta1Grad, d_theta2Grad, 
                    d_tempMatrix );

      floatType_t alpha = -lambda;

/* update the weights with the newly calculated gradients */

      checkCUBLAS( cublasSaxpy( cublasHandle,
                            theta1Rows*theta1Cols,
                            &alpha,
                            d_theta1Grad, 1,
                            d_theta1, 1 ) ); 

      checkCUBLAS( cublasSaxpy( cublasHandle,
                            theta2Rows*theta2Cols,
                            &alpha,
                            d_theta2Grad, 1,
                            d_theta2, 1 ) ); 
    } /* end for */ 
 
    iter++;
    printf("|");
    fflush(stdout);
    if( iter % 72 == 0 ) printf("\n");
  } /* end while */

/* copy theta1 and theta2 back to the host to use for prediction later */

  checkCUDA( cudaMemcpy( theta1, d_theta1,
                         sizeof(floatType_t)*theta1Rows*theta1Cols,
                         cudaMemcpyDeviceToHost ) );
  checkCUDA( cudaMemcpy( theta2, d_theta2,
                         sizeof(floatType_t)*theta2Rows*theta2Cols,
                         cudaMemcpyDeviceToHost ) );

//  printf("\nFinal cost value                      %.3e\n",cost);
  printf("\n");
  checkCUDA( cudaFree( d_tempMatrix ) );
  checkCUDA( cudaFree( d_X ) );
  checkCUDA( cudaFree( d_Y ) );
  checkCUDA( cudaFree( d_theta1 ) );
  checkCUDA( cudaFree( d_theta2 ) );
  checkCUDA( cudaFree( d_theta1Grad ) );
  checkCUDA( cudaFree( d_theta2Grad ) );

} /* end trainNetwork */

void calcGradient( floatType_t       *d_X, 
                   int         const Xexamples, 
                   int         const Xfeatures,
                   floatType_t const *d_theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *d_theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *d_Y, 
                   floatType_t       *cost,
                   floatType_t       *d_theta1Grad,
                   floatType_t       *d_theta2Grad,
                   floatType_t       *d_tempMatrix )
{

  floatType_t *d_z2, *d_a2, *d_a3, *d_delta3, *d_delta2;

/* take tempMatrix and partition it up for use */

  d_z2     = d_tempMatrix;
  d_a2     = &d_z2[Xexamples*(theta1Rows+1)];
  d_a3     = &d_a2[Xexamples*(theta1Rows+1)];
  d_delta2 = &d_a3[Xexamples*(theta2Rows+1)];
  d_delta3 = &d_delta2[Xexamples*(theta1Rows+1)];

  float alpha = 1.0;
  float beta  = 0.0;

  if( sizeof( floatType_t ) == 4 ) 
  {


/* calculate X * theta1 to give z2 */

    checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_T, CUBLAS_OP_T,
                              Xexamples, theta1Rows, theta1Cols,
			      &alpha, d_X, Xfeatures,
                              d_theta1, theta1Rows, &beta,
                              &d_z2[INDX(0,1,Xexamples)], Xexamples ) );                              
/* copy z2 into a2 */

    checkCUDA( cudaMemcpy( d_a2, d_z2, 
                           sizeof(floatType_t) * Xexamples * (theta1Rows+1),
                           cudaMemcpyDeviceToDevice ) );

/* calculate a2 = sigmoid(z2), the activation */

    dim3 threads1(256,1,1);
    dim3 blocks1( Xexamples*(theta1Rows+1)/threads1.x + 1, 1, 1);
    k_sigmoid_f<<< blocks1, threads1 >>>( d_a2, Xexamples*(theta1Rows+1) );
    checkKERNEL()
  
  } /* end if */
  else
  {
  } /* end else */  

/* add a 1.0 to the beginning of each a2 vector for bias term */

  initOne<<< Xexamples/256 + 1, 256 >>>( Xexamples, d_a2 );
  checkKERNEL()


  if( sizeof( floatType_t ) == 4 )
  {

/* calculated z3 = a2 * theta2.  put in a3 array space since we don't 
   need z3 for anything else */

    checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              Xexamples, theta2Rows, theta2Cols,
			      &alpha, d_a2, Xexamples,
                              d_theta2, theta2Rows, &beta,
                              d_a3, Xexamples ) );                              

/* calculate a3 = sigmoid(z3), the activation */

    dim3 threads1(256,1,1);
    dim3 blocks1( Xexamples*(theta2Rows+1)/threads1.x + 1, 1, 1);
    k_sigmoid_f<<< blocks1, threads1 >>>( d_a3, Xexamples*(theta2Rows+1) );
    checkKERNEL()

  } /* end if */
  else
  { 
  } /* end else */

/* enable the following code if you wish to calculate the forward cost 
   not strictly necessary to generate the gradients
   WARNING THIS IS BROKEN RIGHT NOW ON THE GPU!!!
*/
#if 0
  floatType_t jTemp = 0.0;
  for( int row = 0; row < Xexamples; row++ )
  {
    memset( yTemp, 0, sizeof(floatType_t) * 11 ); 
    yTemp[  (int)Y[row]  ] = (floatType_t) 1.0;
    for( int j = 1; j <= theta2Rows; j++ )
    {
      jTemp += -log( a3[INDX(row,j-1,Xexamples)] ) * yTemp[j] 
             - ( log( (floatType_t) 1.0 - a3[INDX(row,j-1,Xexamples)] ) * 
                 ( (floatType_t) 1.0 - yTemp[j] ) ) ;
    } /* end for */
  } /* end for */

  jTemp /= (floatType_t) Xexamples;
  *cost = jTemp;
#endif

  checkCUDA( cudaMemset( d_delta3, 0, sizeof(floatType_t)*11*Xexamples ) );

/* set delta3 to be the difference between a3 and y, the calculated versus
   the actual values 
*/

  setDelta3Vec<<< Xexamples/256+1, 256 >>>( d_delta3, d_Y, d_a3, Xexamples );
  checkKERNEL()

  if( sizeof( floatType_t ) == 4 )
  {

/* calculated delta2 = theta2 * delta3 */

    checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              theta2Cols, Xexamples, theta2Rows,
			      &alpha, (float *)d_theta2, theta2Rows,
                              (float *)&d_delta3[1], 11, &beta,
                              (float *)d_delta2, theta1Rows+1 ) );   

/* calculate the sigmoid gradient of z2 */

   dim3 threads(256,1,1);
   dim3 blocks(Xexamples*(theta1Rows+1)+1/threads.x,1,1);

   k_sigmoidGradient_f<<< blocks, threads >>>( d_z2, Xexamples*(theta1Rows+1) );
   checkKERNEL()

/* update delta2 with the sigmoid gradient of z2 */

   dim3 t1(32,32,1); 
   dim3 b1((theta1Rows+1)/t1.x + 1, Xexamples/t1.y + 1, 1 );

   k_updateDelta2<<< b1, t1 >>>( d_delta2, d_z2, Xexamples, 
                                         theta1Rows+1 );
   checkKERNEL()

   floatType_t recip = (floatType_t) 1.0 / (floatType_t) Xexamples;

/* calculate theta1Grad = delta2 * X */

   checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              theta1Rows, theta1Cols, Xexamples,
			      &recip, (float *)&d_delta2[1], theta1Rows+1,
                              (float *)d_X, Xfeatures, 
                              &beta, (float *)d_theta1Grad, theta1Rows ) );   

/* calculate theta2Grad = delta3 * a2 */

   checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              theta2Rows, theta2Cols, Xexamples,
			      &recip, (float *)&d_delta3[1], 11,
                              (float *)d_a2, Xexamples, 
                              &beta, (float *)d_theta2Grad, theta2Rows ) );   

  } /* end if */
  else
  {
  } /* end else */
} /* end calcGradient */

void predict(floatType_t       *X, 
             int         const Xexamples, 
             int         const Xfeatures,
             floatType_t const *theta1, 
             int         const theta1Rows,
             int         const theta1Cols,
             floatType_t const *theta2, 
             int         const theta2Rows,
             int         const theta2Cols,
             int               *predictVector)
{

  floatType_t *tempMatrix, *z2, *a2, *a3;

/* add the bias term to the X training set data */
 
  for( int i = 0; i < Xexamples; i++ ) 
    X[INDX(0,i,Xfeatures)] = (floatType_t) 1.0;

/* malloc the temp space */

  tempMatrix = (floatType_t *) malloc( sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + 
                                 Xexamples * (theta1Rows+1) +
                                 Xexamples * (theta2Rows+1) ) );

/* carve up the temp space */

  z2 = tempMatrix;
  a2 = &z2[INDX(Xexamples,theta1Rows,Xexamples)];
  a3 = &a2[INDX(Xexamples,theta1Rows+1,Xexamples)];

  if( sizeof( floatType_t ) == 4 ) 
  {

/* calculate z2 */

    cblas_sgemm( CblasColMajor, CblasTrans, CblasTrans,
                 Xexamples, theta1Rows, theta1Cols,
                 1.0f, (float *) X, Xfeatures,
                 (float *) theta1, theta1Rows, 0.0f,
                 (float *) &z2[INDX(0,1,Xexamples)], Xexamples );

/* calculate a2 */

    for( int j = 1; j < theta1Rows+1; j++ )
      for( int i = 0; i < Xexamples; i++ )
        a2[INDX(i,j,Xexamples)] = 
          sigmoid_f( z2[INDX(i,j,Xexamples)] );
  } /* end if */
  else
  {
  } /* end else */  

/* add the bias term to a2 */

  for( int i = 0; i < Xexamples; i++ ) 
    a2[INDX(i,0,Xexamples)] = (floatType_t) 1.0;

  if( sizeof( floatType_t ) == 4 )
  {

/* calculate z3 */

    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );

/* calculate a3 */

    for( int j = 0; j < theta2Rows; j++ )
      for( int i = 0; i < Xexamples; i++ )
        a3[INDX(i,j,Xexamples)] = 
          sigmoid_f( a3[INDX(i,j,Xexamples)] );
  } /* end if */
  else
  { 
  } /* end else */

/* use a3 to populate the prediction vector.  each element will be a 
   digit between one and ten, which is the predicted value of the image
*/

  for( int row = 0; row < Xexamples; row++ )
  {
    floatType_t max = -99.0;
    int         idx = -10;
    for( int i = 0; i < 10; i++ )
    {
      if( a3[INDX(row,i,Xexamples)] > max )
      {
        max = a3[INDX(row,i,Xexamples)];
        idx = i+1;
      } /* end if */
    } /* end for i */
    predictVector[row] = idx;
  } /* end row */

  free(tempMatrix); 
} /* end predict */ 

void readCommandLineArgs( int    argc, 
                          char   *argv[],
                          float  *learningRate,
                          int    *batchSize,
                          int    *iterations,
                          int    *sizeHiddenLayer )
{
/* read command line input */
  switch( argc )
  {
    case 1:
      *learningRate = 0.3;
      *batchSize = 50;
      *iterations = 1;
      *sizeHiddenLayer = 25;
      break;
    case 2:
      if( strcmp( argv[1],"-h" ) == 0 )
      {
        printf("Usage: ./x.nn -h for this message\n");
        printf("Usage: ./x.nn <learningRate:float> <batchSize:int> <iterations:int> <hiddenLayerSize:int>\n");
        exit(911);
      } /* end for */
      break;
    case 5:
      *learningRate = atof( argv[1] );
      if( *learningRate == 0.0f )
      {
        printf("Invalid learning rate %s\n", argv[1] );
        *learningRate = 0.3;
        printf("Defaulting to %e\n", *learningRate );
      } /* end if */

      *batchSize = atoi( argv[2] );
      if( *batchSize <= 0 )
      {
        printf("Invalid batchSize %s\n", argv[2] );
        *batchSize = 50;
        printf("Defaulting to %d\n",*batchSize );
      } /* end if */

      *iterations = atoi( argv[3] );
      if( *iterations <= 0 )
      {
        printf("Invalid iteration size %s\n", argv[3] );
        *iterations = 1;
        printf("Defaulting to %d\n",*iterations);
      } /* end if */

      *sizeHiddenLayer = atoi( argv[4] );
      if( *sizeHiddenLayer <= 0 )
      {
        printf("Invalid hidden layer size %s\n", argv[4] );
        *sizeHiddenLayer = 25;
        printf("Defaulting to %d\n",*sizeHiddenLayer );
      } /* end if */
      break;
    default:
      printf("Undefined command-line args\n");
      printf("Usage: ./x.nn -h for this message\n");
      printf("Usage: ./x.nn <learningRate:float> <batchSize:int> <iterations:int> <hiddenLayerSize:int>\n");
      exit(911);
      break;

  } /* end switch */

/* print some initial stuff */
  printf("Learning rate lambda is               %.3e\n",*learningRate);
  printf("Batchsize is                          %d\n",*batchSize);
  printf("Number of iterations is               %d\n",*iterations);
  printf("Hidden Layer Size is                  %d\n",*sizeHiddenLayer);


} /* end readCommandLineArgs */


void readMatrixFromFile( char *fileName, 
                         float *matrix, 
                         int const rows, 
                         int const cols,
                         int const ld )
{
  FILE *ifp;

  ifp = fopen( fileName, "r" );

  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening file %s\n", fileName);
    exit(911);
  } /* end if */

  for( int col = 0; col < cols; col++ )
  {
    for( int row = 0; row < rows; row++ )
    {
      if( !fscanf( ifp, "%f", 
          &matrix[ INDX( row, col, ld ) ] ) )
      {
        fprintf(stderr,"error reading training matrix file \n");
        exit(911);
      } /* end if */
    } /* end for row */
  } /* end for col */

  fclose(ifp);
  return;
} /* end readMatrixFromFile */

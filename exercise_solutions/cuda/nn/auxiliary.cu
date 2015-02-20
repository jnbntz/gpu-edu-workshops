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
#include <cudnn.h>
#include <cublas_v2.h>
#include "headers.h"

cudnnHandle_t cudnnHandle;
cudnnTensorDescriptor_t srcTensorDesc, destTensorDesc;
cublasHandle_t cublasHandle;

__global__ void setVals( int rows, floatType_t *array )
{
  for( int i = 0; i < rows; i++ )
    array[i] = (floatType_t)i;
} /* end setVAls */

__global__ void initOne( int size, floatType_t *array )
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if( tid < size )
    array[tid] = (floatType_t) 1.0;
  return;
} /* end initOne */

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
  floatType_t *theta1Grad, *theta2Grad, *tempMatrix;



  checkCUDNN( cudnnCreate( &cudnnHandle ) );
  checkCUDNN( cudnnCreateTensorDescriptor( &srcTensorDesc ) );
  checkCUDNN( cudnnCreateTensorDescriptor( &destTensorDesc ) );
  checkCUBLAS( cublasCreate( &cublasHandle ) );

  theta1Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta1Rows * theta1Cols );

  theta2Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta2Rows * theta2Cols );

  tempMatrix = (floatType_t *) malloc( sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + 
                                 Xexamples * (theta1Rows+1) +
                                 Xexamples * (theta2Rows+1) + 
                                 theta2Cols + 11) );

  for( int i = 0; i < Xexamples; i++ ) 
    X[INDX(0,i,Xfeatures)] = (floatType_t) 1.0;

#if 1
/* stochastic gradient descent */
  int iter = 0;
//  int batchSize = 64;

//  printf("Learning rate Lambda is %f\n",lambda);
 // printf("Batchsize is %d\n",batchSize);

  while(iter < iterations )
  {
//  for( int i = 0; i < 500; i++ )
 // {
    for( int j = 0; j < Xexamples; j+=batchSize )
    {
   //   int j = (int) ((double(rand()) / (double(RAND_MAX) + 1.0))*5000);
      
      costFunction( &X[INDX(0,j,Xfeatures)], batchSize, Xfeatures,
                    theta1, theta1Rows, theta1Cols, 
                    theta2, theta2Rows, theta2Cols,
                    &Y[j],
                    &cost, theta1Grad, theta2Grad, 
                    tempMatrix );

//      printf("iter %d j %d cost is %.3e val %f\n",iter,j,cost,Y[j]);

      for( int i = 0; i < theta1Rows*theta1Cols; i++ )
        theta1[i] -= lambda * theta1Grad[i];

      for( int i = 0; i < theta2Rows*theta2Cols; i++ )
        theta2[i] -= lambda * theta2Grad[i];
//      printf("j %d val %f\n",j,Y[j]);
//     exit(911);
    } 
 // } /* end for i */
  iter++;
    printf("|");
    fflush(stdout);
    if( iter % 72 == 0 ) printf("\n");
  } /* end while */
#endif
#if 0
/* gradient descent algorithm */

  int iter = 0;

  while( iter < 20 )
  {

  costFunction( X, Xexamples, Xfeatures,
                theta1, theta1Rows, theta1Cols, 
                theta2, theta2Rows, theta2Cols,
                Y,
                &cost, theta1Grad, theta2Grad );

  printf("iter %d cost is %.3e\n",iter,cost);

  for( int i = 0; i < theta1Rows*theta1Cols; i++ )
    theta1[i] -= lambda * theta1Grad[i];

  for( int i = 0; i < theta2Rows*theta2Cols; i++ )
    theta2[i] -= lambda * theta2Grad[i];

    iter++;

//    printf("|");
 //   fflush(stdout);
  //  if( iter % 72 == 0 ) printf("\n");
  } /* end while */
#endif
  printf("\nFinal cost value                      %.3e\n",cost);
  free(tempMatrix);
  free(theta1Grad);
  free(theta2Grad);

} /* end trainNetwork */

void costFunction( floatType_t       *X, 
                   int         const Xexamples, 
                   int         const Xfeatures,
                   floatType_t const *theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y, 
                   floatType_t       *cost,
                   floatType_t       *theta1Grad,
                   floatType_t       *theta2Grad,
                   floatType_t       *tempMatrix )
{

  floatType_t *z2, *a2, *a3;
  floatType_t *yTemp;
  floatType_t *delta2;

/* offset the pointers in the scratch memory */

  z2 = tempMatrix;
  a2 = &z2[INDX(Xexamples,theta1Rows,Xexamples)];
  a3 = &a2[INDX(Xexamples,theta1Rows+1,Xexamples)];
  yTemp = &a3[INDX(Xexamples,theta2Rows+1,Xexamples)];
  delta2 = &yTemp[11];

  floatType_t *d_tempMatrix;
  CUDA_CALL( cudaMalloc( &d_tempMatrix, sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + 
                                 Xexamples * (theta1Rows+1) +
                                 Xexamples * (theta2Rows+1) + 
                                 theta2Cols + 11) ) ) ;

  floatType_t *d_z2, *d_a2, *d_a3, *d_yTemp, *d_delta2;
  d_z2 = d_tempMatrix;
  d_a2 = &d_z2[INDX(Xexamples,theta1Rows,Xexamples)];
  d_a3 = &d_a2[INDX(Xexamples,theta1Rows+1,Xexamples)];
  d_yTemp = &d_a3[INDX(Xexamples,theta2Rows+1,Xexamples)];
  d_delta2 = &d_yTemp[11];


  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         Xexamples,
                                         theta1Rows+1,
                                         1,1) );

  checkCUDNN( cudnnSetTensor4dDescriptor(destTensorDesc,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         Xexamples,
                                         theta1Rows+1,
                                         1,1) );


  floatType_t *d_X;
  CUDA_CALL( cudaMalloc( &d_X,
          sizeof(floatType_t)*Xexamples*Xfeatures ) );

  floatType_t *d_theta1;
  CUDA_CALL( cudaMalloc( &d_theta1, 
          sizeof(floatType_t) * theta1Rows * theta1Cols ) );

  floatType_t *d_theta2;
  CUDA_CALL( cudaMalloc( &d_theta2, 
          sizeof(floatType_t) * theta2Rows * theta2Cols ) );

  float alpha = 1.0;
  float beta  = 0.0;

#if 1
  if( sizeof( floatType_t ) == 4 ) 
  {
//    cblas_sgemm( CblasColMajor, CblasTrans, CblasTrans,
 //                Xexamples, theta1Rows, theta1Cols,
  //               1.0f, (float *) X, Xfeatures,
   //              (float *) theta1, theta1Rows, 0.0f,
    //             (float *) &z2[INDX(0,1,Xexamples)], Xexamples );

//    printHost(100,1,&z2[INDX(0,1,Xexamples)] );

#if 0
    for( int i = Xexamples; i < Xexamples*(theta1Rows+1); i++ )
      a2[i] = sigmoid_f( z2[i] );
#endif
#if 1
    CUDA_CALL( cudaMemcpy( d_X, X,
                           sizeof(floatType_t)*Xexamples*Xfeatures,
                           cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_theta1, theta1,
                           sizeof(floatType_t)*theta1Rows*theta1Cols,
                           cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_theta2, theta2,
                           sizeof(floatType_t)*theta2Rows*theta2Cols,
                           cudaMemcpyHostToDevice ) );
//    CUDA_CALL( cudaMemcpy( d_srcData, z2, 
 //                          sizeof(floatType_t)*Xexamples*(theta1Rows+1),
  //                         cudaMemcpyHostToDevice ) );
//    printf("rows %d cols %d\n",theta1Rows,theta1Cols);


//    printHost(Xfeatures*Xexamples,1,X);
 //   printKernel<<<1,1>>>( Xfeatures*Xexamples, 1, d_X );
  //  CUDA_CHECK()
   // CUDA_CALL( cudaDeviceSynchronize() );

    checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_T, CUBLAS_OP_T,
                              Xexamples, theta1Rows, theta1Cols,
			      &alpha, d_X, Xfeatures,
                              d_theta1, theta1Rows, &beta,
                              &d_z2[INDX(0,1,Xexamples)], Xexamples ) );                              
    CUDA_CALL( cudaMemcpy( z2, d_z2, 
                           sizeof(floatType_t)*Xexamples*(theta1Rows+1),
                           cudaMemcpyDeviceToHost ) );
    
//    setVals<<<1,1>>>(10,d_srcData );
 //   printKernel<<<1,1>>>( 5, 1, &d_srcData[5] );
  //  CUDA_CHECK()
   // CUDA_CALL( cudaDeviceSynchronize() );
                                

    checkCUDNN( cudnnActivationForward( cudnnHandle,
                                        CUDNN_ACTIVATION_SIGMOID,
                                        &alpha,
                                        srcTensorDesc, d_z2,
                                        &beta,
                                        destTensorDesc, d_a2 ) );
#endif
//exit(911);

  } /* end if */
  else
  {
  } /* end else */  

#if 1
    initOne<<< Xexamples/256 + 1, 256 >>>( Xexamples, d_a2 );
    CUDA_CHECK()
    CUDA_CALL( cudaDeviceSynchronize() );
    CUDA_CALL( cudaMemcpy( a2, d_a2, 
                           sizeof(floatType_t)*Xexamples*(theta1Rows+1),
                           cudaMemcpyDeviceToHost ) );
#endif
#if 0
  for( int i = 0; i < Xexamples; i++ ) 
    a2[INDX(i,0,Xexamples)] = (floatType_t) 1.0;
#endif

  if( sizeof( floatType_t ) == 4 )
  {
#if 0
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );
#endif
#if 1
    checkCUBLAS( cublasSgemm( cublasHandle, 
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              Xexamples, theta2Rows, theta2Cols,
			      &alpha, d_a2, Xexamples,
                              d_theta2, theta2Rows, &beta,
                              d_a3, Xexamples ) );                              

  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         Xexamples,
                                         theta2Rows,
                                         1,1) );

  checkCUDNN( cudnnSetTensor4dDescriptor(destTensorDesc,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         Xexamples,
                                         theta2Rows,
                                         1,1) );

    checkCUDNN( cudnnActivationForward( cudnnHandle,
                                        CUDNN_ACTIVATION_SIGMOID,
                                        &alpha,
                                        srcTensorDesc, d_a3,
                                        &beta,
                                        destTensorDesc, d_a3 ) );
    CUDA_CALL( cudaMemcpy( a3, d_a3, 
                           sizeof(floatType_t)*theta2Rows*Xexamples,
                           cudaMemcpyDeviceToHost ) );

#endif
#if 0
      for( int i = 0; i < theta2Rows*Xexamples; i++ )
        a3[i] = sigmoid_f( a3[i] );
#endif

  } /* end if */
  else
  { 
  } /* end else */

/* enable the following code if you wish to calculate the forward cost 
   not strictly necessary to generate the gradients
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
#endif
#if 1
  floatType_t *delta3;
  delta3 = yTemp;


  memset( theta1Grad, 0, sizeof(floatType_t) * theta1Rows * theta1Cols );

  memset( theta2Grad, 0, sizeof(floatType_t) * theta2Rows * theta2Cols );

  for( int row = 0; row < Xexamples; row++ )
  { 
    memset( delta3, 0, sizeof( floatType_t) * 11 );
    delta3[ (int) Y[row] ] = (floatType_t) 1.0;
#if 1
    for( int j = 0; j < 10; j++ ) 
    {
      delta3[j+1] = a3[INDX(row,j,Xexamples)] - delta3[j+1];
    } /* end for j */

    if( sizeof( floatType_t ) == 4 )
    {
      cblas_sgemv( CblasColMajor, CblasTrans,
                 theta2Rows, theta2Cols,
                 1.0f, theta2, theta2Rows, 
                 &delta3[1], 1, 0.0f,
                 delta2, 1 );

      for( int j = 1; j <= theta1Rows; j++ )
      {
        delta2[j] *= sigmoidGradient_f( z2[INDX(row,j,Xexamples)] );
      } /* end for */
    } /* end if */
    else
    { 
    } /* end else */
#endif
#if 1
    for( int j = 0; j < theta1Cols; j++ )
    {
      for( int i = 0; i < theta1Rows; i++ )
      {
        theta1Grad[INDX(i,j,theta1Rows)] += 
          ( delta2[i+1] * X[INDX(j,row,Xfeatures)] );
      } /* end for i */    
    } /* end for j */

    for( int j = 0; j < theta2Cols; j++ )
    {
      for( int i = 0; i < theta2Rows; i++ )
      {
        theta2Grad[INDX(i,j,theta2Rows)] +=
          ( delta3[i+1] * a2[INDX(row,j,Xexamples)] );
      } /* end for i */
    } /* end for j */
#endif
  } /* end for row */

  floatType_t recip = (floatType_t) 1.0 / (floatType_t) Xexamples;

//  for( int j = 0; j < theta1Cols; j++ )
 // {
  //  for( int i = 0; i < theta1Rows; i++ )
   // {
    //  theta1Grad[INDX(i,j,theta1Rows)] *= recip;
  //  } /* end for i */    
//  } /* end for j */

  for( int i = 0; i < theta1Rows*theta1Cols; i++ )
    theta1Grad[i] *= recip;

//  for( int j = 0; j < theta2Cols; j++ )
 // {
  //  for( int i = 0; i < theta2Rows; i++ )
   // {
    //  theta2Grad[INDX(i,j,theta2Rows)] *= recip;
  //  } /* end for i */
//  } /* end for j */
  for( int i = 0; i < theta2Cols*theta2Rows; i++ )
    theta2Grad[i] *= recip;
#endif

  CUDA_CALL( cudaFree( d_tempMatrix ) );
  CUDA_CALL( cudaFree( d_X ) );
  CUDA_CALL( cudaFree( d_theta1 ) );
  CUDA_CALL( cudaFree( d_theta2 ) );
} /* end costFunction */

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
 
  for( int i = 0; i < Xexamples; i++ ) 
    X[INDX(0,i,Xfeatures)] = (floatType_t) 1.0;

  tempMatrix = (floatType_t *) malloc( sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + 
                                 Xexamples * (theta1Rows+1) +
                                 Xexamples * (theta2Rows+1) ) );

  z2 = tempMatrix;
  a2 = &z2[INDX(Xexamples,theta1Rows,Xexamples)];
  a3 = &a2[INDX(Xexamples,theta1Rows+1,Xexamples)];

  if( sizeof( floatType_t ) == 4 ) 
  {
    cblas_sgemm( CblasColMajor, CblasTrans, CblasTrans,
                 Xexamples, theta1Rows, theta1Cols,
                 1.0f, (float *) X, Xfeatures,
                 (float *) theta1, theta1Rows, 0.0f,
                 (float *) &z2[INDX(0,1,Xexamples)], Xexamples );
    for( int j = 1; j < theta1Rows+1; j++ )
      for( int i = 0; i < Xexamples; i++ )
        a2[INDX(i,j,Xexamples)] = 
          sigmoid_f( z2[INDX(i,j,Xexamples)] );
  } /* end if */
  else
  {
  } /* end else */  



  for( int i = 0; i < Xexamples; i++ ) 
    a2[INDX(i,0,Xexamples)] = (floatType_t) 1.0;

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );
    for( int j = 0; j < theta2Rows; j++ )
      for( int i = 0; i < Xexamples; i++ )
        a3[INDX(i,j,Xexamples)] = 
          sigmoid_f( a3[INDX(i,j,Xexamples)] );
  } /* end if */
  else
  { 
  } /* end else */

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

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
#include "headers.h"

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

/* malloc the gradient arrays */

  theta1Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta1Rows * theta1Cols );

  theta2Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta2Rows * theta2Cols );

/* malloc the array for temp space */

  tempMatrix = (floatType_t *) malloc( sizeof(floatType_t) *
                               ( Xexamples * (theta1Rows+1) + //z2
                                 Xexamples * (theta1Rows+1) + //a2
                                 Xexamples * (theta2Rows+1) + //a3
                                 Xexamples * (theta1Rows+1) + //delta2
                                 Xexamples * 11) );           //delta3

  for( int i = 0; i < Xexamples; i++ ) 
    X[INDX(0,i,Xfeatures)] = (floatType_t) 1.0;

/* stochastic gradient descent
   in our case stochastic because the data is already scrambled */

  int iter = 0;

  while(iter < iterations )
  {

/* for loop over the batch size */

    for( int j = 0; j < Xexamples; j+=batchSize )
    {
      int tempBatchSize = min( batchSize, Xexamples - j );      

/* bulk of computation here */

      calcGradient( &X[INDX(0,j,Xfeatures)], tempBatchSize, Xfeatures,
                    theta1, theta1Rows, theta1Cols, 
                    theta2, theta2Rows, theta2Cols,
                    &Y[j],
                    &cost, theta1Grad, theta2Grad, 
                    tempMatrix );

/* update the weights with the newly calculated gradients */

      for( int i = 0; i < theta1Rows*theta1Cols; i++ )
        theta1[i] -= lambda * theta1Grad[i];

      for( int i = 0; i < theta2Rows*theta2Cols; i++ )
        theta2[i] -= lambda * theta2Grad[i];
    } /* end for */

    iter++;
    printf("|");
    fflush(stdout);
    if( iter % 72 == 0 ) printf("\n");
  } /* end while */
  printf("\n");
  free(tempMatrix);
  free(theta1Grad);
  free(theta2Grad);

} /* end trainNetwork */

void calcGradient( floatType_t       *X, 
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
  floatType_t *delta3;
  floatType_t *delta2;

/* take tempMatrix and partition it up for use */

  z2     = tempMatrix;
  a2     = &z2[Xexamples*(theta1Rows+1)];
  a3     = &a2[Xexamples*(theta1Rows+1)];
  delta2 = &a3[Xexamples*(theta2Rows+1)];
  delta3  = &delta2[Xexamples*(theta1Rows+1)];

  if( sizeof( floatType_t ) == 4 ) 
  {

/* calculate X * theta1 to give z2 */

    cblas_sgemm( CblasColMajor, CblasTrans, CblasTrans,
                 Xexamples, theta1Rows, theta1Cols,
                 1.0f, (float *) X, Xfeatures,
                 (float *) theta1, theta1Rows, 0.0f,
                 (float *) &z2[INDX(0,1,Xexamples)], Xexamples );

/* calculate a2 = sigmoid(z2), the activation */

    for( int i = Xexamples; i < Xexamples*(theta1Rows+1); i++ )
      a2[i] = sigmoid_f( z2[i] );

  } /* end if */
  else
  {
  } /* end else */  

/* add a 1.0 to the beginning of each a2 vector for bias term */

  for( int i = 0; i < Xexamples; i++ ) 
    a2[INDX(i,0,Xexamples)] = (floatType_t) 1.0;

  if( sizeof( floatType_t ) == 4 )
  {

/* calculated z3 = a2 * theta2.  put in a3 array space since we don't
   need z3 for anything else */

    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );

/* calculate a3 = sigmoid(z3), the activation */

      for( int i = 0; i < theta2Rows*Xexamples; i++ )
        a3[i] = sigmoid_f( a3[i] );

  } /* end if */
  else
  { 
  } /* end else */

/* enabled the following code if you wish to calculate the forward cost 
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

#if 1
  for( int row = 0; row < Xexamples; row++ )
  { 
    memset( &delta3[INDX(0,row,11)], 0, sizeof( floatType_t) * 11 );

/* set delta3 to be the difference between a3 and y, the calculated versus
   the actual values
*/

    delta3[INDX((int)Y[row],row,11)] = (floatType_t) 1.0;
    for( int j = 0; j < 10; j++ ) 
    {
      delta3[INDX(j+1,row,11)] = a3[INDX(row,j,Xexamples)] 
                               - delta3[INDX(j+1,row,11)];
    } /* end for j */
  } /* end for */

  if( sizeof( floatType_t ) == 4 )
  {

/* calculated delta2 = theta2 * delta3 */

    cblas_sgemm( CblasColMajor, CblasTrans, CblasNoTrans,
                 theta2Cols, Xexamples, theta2Rows,
                 1.0f, theta2, theta2Rows,
                 &delta3[1],11, 0.0f,
                 delta2,theta1Rows+1);

/* calculate the sigmoid gradient of z2 */

   for( int i = 0; i < Xexamples*(theta1Rows+1); i++ )
     z2[i] = sigmoidGradient_f( z2[i] );
   
/* update delta2 with the sigmoid gradient of z2 */

    for( int row = 0; row < Xexamples; row++ )
    { 
      for( int j = 0; j < theta1Rows+1; j++ )
      {
        delta2[INDX(j,row,theta1Rows+1)] *= z2[INDX(row,j,Xexamples)];
      } /* end for */
    } /* end for */
  } /* end if */
  else
  { 
  } /* end else */

  floatType_t recip = (floatType_t) 1.0 / (floatType_t) Xexamples;

/* calculate theta1Grad = delta2 * X */

  cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
               theta1Rows, theta1Cols, Xexamples,
               recip, (float *) &delta2[1], theta1Rows+1,
               X, Xfeatures,
               0.0f, (float *) theta1Grad, theta1Rows );

/* calculate theta2Grad = delta3 * a2 */

  cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
               theta2Rows, theta2Cols, Xexamples,
               recip, (float *) &delta3[1], 11,
               (float *) a2, Xexamples, 0.0f,
               (float *) theta2Grad, theta2Rows );


#endif
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

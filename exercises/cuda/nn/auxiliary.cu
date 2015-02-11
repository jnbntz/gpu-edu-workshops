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

void trainNetwork( floatType_t       *X, 
                   int         const Xexamples, 
                   int         const Xfeatures,
                   floatType_t       *theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t       *theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y )
{
  floatType_t lambda = 0.3;
  floatType_t cost;
  floatType_t *theta1Grad, *theta2Grad;

  theta1Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta1Rows * theta1Cols );

  theta2Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
                                theta2Rows * theta2Cols );

#if 1
/* stochastic gradient descent */
  int iter = 0;
  while(iter < 20 )
  {
  for( int i = 0; i < 500; i++ )
  {
    for( int j = i; j < 5000; j+=500 )
    {
   //   int j = (int) ((double(rand()) / (double(RAND_MAX) + 1.0))*5000);
      
      costFunction( &X[INDX(0,j,Xfeatures)], 1, Xfeatures,
                    theta1, theta1Rows, theta1Cols, 
                    theta2, theta2Rows, theta2Cols,
                    &Y[j],
                    &cost, theta1Grad, theta2Grad );

//      printf("iter %d j %d cost is %.3e val %f\n",iter,j,cost,Y[j]);

      for( int i = 0; i < theta1Rows*theta1Cols; i++ )
        theta1[i] -= lambda * theta1Grad[i];

      for( int i = 0; i < theta2Rows*theta2Cols; i++ )
        theta2[i] -= lambda * theta2Grad[i];
//      printf("j %d val %f\n",j,Y[j]);
    } 
  } /* end for i */
  iter++;
    printf("|");
    fflush(stdout);
    if( iter % 72 == 0 ) printf("\n");
  } /* end while */
#endif
#if 0
/* gradient descent algorithm */

  int iter = 0;

  while( iter < 400 )
  {

  costFunction( X, Xexamples, Xfeatures,
                theta1, theta1Rows, theta1Cols, 
                theta2, theta2Rows, theta2Cols,
                Y,
                &cost, theta1Grad, theta2Grad );

//  printf("iter %d cost is %.3e\n",iter,cost);

  for( int i = 0; i < theta1Rows*theta1Cols; i++ )
    theta1[i] -= lambda * theta1Grad[i];

  for( int i = 0; i < theta2Rows*theta2Cols; i++ )
    theta2[i] -= lambda * theta2Grad[i];

    iter++;

    printf("|");
    fflush(stdout);
    if( iter % 72 == 0 ) printf("\n");
  } /* end while */
#endif
  printf("\nfinal cost value %.3e\n",cost);

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
                   floatType_t       *theta2Grad )
{

  floatType_t *tempMatrix, *z2, *a2, *a3;
//  floatType_t *theta1Grad, *theta2Grad;

//  printf("Xrows %d Xcols %d\n",XRows,XCols);
 // printf("t1row %d t1col %d\n",theta1Rows,theta1Cols);
  //printf("t2row %d t2col %d\n",theta2Rows,theta2Cols);

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
//                 (float *) &tempMatrix[INDX(0,1,XRows)], XRows );
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

//  a3 = &tempMatrix[INDX(0,theta2Cols+1,XRows)];

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );
//                 (float *) &tempMatrix[INDX(0,theta2Cols+1,XRows)], XRows );
    for( int j = 0; j < theta2Rows; j++ )
      for( int i = 0; i < Xexamples; i++ )
        a3[INDX(i,j,Xexamples)] = 
          sigmoid_f( a3[INDX(i,j,Xexamples)] );
  } /* end if */
  else
  { 
  } /* end else */

//  for( int i = 0; i < theta2Rows; i++ )
 //   printf("col %d val %f\n",i,a3[INDX(4999,i,XRows)] );
//    printf("col %d val %f\n",i,tempMatrix[INDX(4999,theta2Cols+1+i,XRows)] );

  


//  for( int i = 0; i < theta2Rows; i++ )
 //   printf("col %d val %e\n",i,a3[INDX(4999,i,XRows)] );

  floatType_t yTemp[11];
  floatType_t jTemp = 0.0;

  for( int row = 0; row < Xexamples; row++ )
  {
    memset( yTemp, 0, sizeof(floatType_t) * 11 ); 
//   printf("row %d Y %f %d\n",row,Y[row],(int) Y[row] );
    yTemp[  (int)Y[row]  ] = (floatType_t) 1.0;
    for( int j = 1; j <= theta2Rows; j++ )
    {
//      printf("j %d val %f\n",j,-log(a3[INDX(row,j-1,XRows)])*yTemp[j]);
      jTemp += -log( a3[INDX(row,j-1,Xexamples)] ) * yTemp[j] 
             - ( log( (floatType_t) 1.0 - a3[INDX(row,j-1,Xexamples)] ) * 
                 ( (floatType_t) 1.0 - yTemp[j] ) ) ;
    } /* end for */
//    printf("row=%d jTemp is %f\n",row,jTemp);
  } /* end for */

  jTemp /= (floatType_t) Xexamples;
//  printf("jTemp is %f %f\n",jTemp, jTemp / (floatType_t)XRows );

  *cost = jTemp;

  floatType_t *tempY, *delta3;
  tempY = (floatType_t *)malloc( sizeof(floatType_t)*11);
  delta3 = tempY;

  floatType_t *delta2;
  delta2 = (floatType_t *)malloc( sizeof(floatType_t) * theta2Cols );

//  theta1Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
 //                               theta1Rows * theta1Cols );


  memset( theta1Grad, 0, sizeof(floatType_t) * theta1Rows * theta1Cols );

//  theta2Grad = (floatType_t *) malloc( sizeof(floatType_t) * 
 //                               theta2Rows * theta2Cols );

  memset( theta2Grad, 0, sizeof(floatType_t) * theta2Rows * theta2Cols );

  for( int row = 0; row < Xexamples; row++ )
  { 
    memset( tempY, 0, sizeof( floatType_t) * 11 );
    tempY[ (int) Y[row] ] = (floatType_t) 1.0;

    for( int j = 0; j < 10; j++ ) 
    {
      tempY[j+1] = a3[INDX(row,j,Xexamples)] - tempY[j+1];
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

    for( int j = 0; j < theta1Cols; j++ )
    {
      for( int i = 0; i < theta1Rows; i++ )
      {
        theta1Grad[INDX(i,j,theta1Rows)] += 
          ( delta2[i+1] * X[INDX(j,row,Xfeatures)] );
//        printf("i %d j %d val %f\n",i,j,theta1Grad[INDX(i,j,theta1Rows)]);
      } /* end for i */    
    } /* end for j */

    for( int j = 0; j < theta2Cols; j++ )
    {
      for( int i = 0; i < theta2Rows; i++ )
      {
        theta2Grad[INDX(i,j,theta2Rows)] +=
          ( delta3[i+1] * a2[INDX(row,j,Xexamples)] );
//        printf("i %d j %d val %e\n",i,j,theta2Grad[INDX(i,j,theta2Rows)]);
      } /* end for i */
    } /* end for j */

  } /* end for row */

  floatType_t recip = (floatType_t) 1.0 / (floatType_t) Xexamples;

  for( int j = 0; j < theta1Cols; j++ )
  {
    for( int i = 0; i < theta1Rows; i++ )
    {
      theta1Grad[INDX(i,j,theta1Rows)] *= recip;
//      printf("i %d j %d val %e\n",i,j,theta1Grad[INDX(i,j,theta1Rows)]);
    } /* end for i */    
//    printf("\n");
  } /* end for j */

  for( int j = 0; j < theta2Cols; j++ )
  {
    for( int i = 0; i < theta2Rows; i++ )
    {
      theta2Grad[INDX(i,j,theta2Rows)] *= recip;
//      printf("i %d j %d val %e\n",i,j,theta2Grad[INDX(i,j,theta2Rows)]);
    } /* end for i */
 //   printf("\n");
  } /* end for j */

} /* end costFunction */

void predict( floatType_t *X, 
                   int const Xexamples, 
                   int const Xfeatures,
                   floatType_t const *theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   int               *predictVector)
{

  floatType_t *tempMatrix, *z2, *a2, *a3;
//  floatType_t *theta1Grad, *theta2Grad;
 
  printf("Xrows %d Xcols %d\n",Xexamples,Xfeatures);
  printf("t1row %d t1col %d\n",theta1Rows,theta1Cols);
  printf("t2row %d t2col %d\n",theta2Rows,theta2Cols);

//  for( int i = 0; i < XRows; i++ ) X[i] = (floatType_t) 1.0;
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
//                 (float *) &tempMatrix[INDX(0,1,XRows)], XRows );
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

//  a3 = &tempMatrix[INDX(0,theta2Cols+1,XRows)];

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 Xexamples, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, Xexamples,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, Xexamples );
//                 (float *) &tempMatrix[INDX(0,theta2Cols+1,XRows)], XRows );
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

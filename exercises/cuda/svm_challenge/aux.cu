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

#define INDX(row,col,ld) (((col) * (ld)) + (row))

void svmTrain( floatType_t const *X, 
               floatType_t const *y, 
               floatType_t const C,
               int const numFeatures, int const numTrainingExamples,
               floatType_t const tol,
               floatType_t *W )
{

/* declare pointers for arrays */
  floatType_t *K, *alphas, *f;

/* declare variables */
  floatType_t bHigh, bLow, eta;
  floatType_t alphaILow=0.0, alphaIHigh=0.0; 
  floatType_t alphaILowPrime=0.0, alphaIHighPrime=0.0;
  int ILow, IHigh;

/* malloc alphas */

  alphas = (floatType_t *) malloc( sizeof(floatType_t) * numTrainingExamples );
  if( alphas == NULL ) fprintf(stderr,"error malloc alphas\n");

/* zero alphas */

  memset( alphas, 0, sizeof(floatType_t) * numTrainingExamples );

/* malloc f */

  f = (floatType_t *) malloc( sizeof(floatType_t) * numTrainingExamples );
  if( f == NULL )
    fprintf(stderr,"error malloc f\n");

/* initialize y */

  for( int i = 0; i < numTrainingExamples; i++ )
    f[i] = -y[i];

/* malloc K, the kernel matrix */

  K = (floatType_t *) malloc( sizeof(floatType_t) * numTrainingExamples *
                              numTrainingExamples );
  if( K == NULL )
    fprintf(stderr,"error malloc K\n");

/* compute the Kernel on every pair of examples.
   K = X * X'
   Wouldn't do this in real life especially if X was really large.  
   For large K we'd just calculate the rows needed on the fly in the
   large loop
*/

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
               numTrainingExamples, numTrainingExamples, numFeatures,
               1.0, (float *)X, numTrainingExamples,
               (float *)X, numTrainingExamples, 0.0,
               (float *)K, numTrainingExamples );
  }
  else
  {
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans,
               numTrainingExamples, numTrainingExamples, numFeatures,
               1.0, (double *)X, numTrainingExamples,
               (double *)X, numTrainingExamples, 0.0,
               (double *)K, numTrainingExamples );
  }

/* start iterative loop */

  while( true )
  {

/* calculate bHigh/bLow/IHigh/ILow */
    calculateBI( f, alphas, y, numTrainingExamples,
                 &bLow, &bHigh, &ILow, &IHigh, C );

/* exit loop once we are below tolerance level */     
    if( bLow <= ( bHigh + ((floatType_t) 2.0 * tol) ) ) 
      break; 

/* grab alpha values */

    alphaILow  = alphas[ILow];
    alphaIHigh = alphas[IHigh];

/* calculate eta */

    eta = K[INDX(IHigh,IHigh,numTrainingExamples)]
        + K[INDX(ILow, ILow, numTrainingExamples)]
        - (floatType_t)2.0 * K[INDX(IHigh,ILow,numTrainingExamples)];

/* calculate new alpha values */

    alphaILowPrime  = alphaILow + ( y[ILow] * ( bHigh - bLow ) ) / eta;
    alphaIHighPrime = alphaIHigh + 
                    y[ILow] * y[IHigh] * ( alphaILow - alphaILowPrime );

/* clip the values */

    CLIP( alphaILowPrime,  (floatType_t) 0.0, C );
    CLIP( alphaIHighPrime, (floatType_t) 0.0, C );

/* update alpha values in the array */

    alphas[ILow]  = alphaILowPrime;
    alphas[IHigh] = alphaIHighPrime;

/* update f array */
    for( int i = 0; i < numTrainingExamples; i++ )
    {
      f[i] = f[i] 
           + ( ( alphaIHighPrime - alphaIHigh ) 
               * y[IHigh] * K[INDX(IHigh,i,numTrainingExamples)] )
           + ( ( alphaILowPrime - alphaILow ) 
               * y[ILow] * K[INDX(ILow,i,numTrainingExamples)] );
    } /* end for i */

  } /* end while */

/* calculate W from alphas */

  if( sizeof( floatType_t ) == 4 )
  {
    for( int i = 0; i < numTrainingExamples; i++ )
      alphas[i] *= y[i];
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
               1, numFeatures, numTrainingExamples,
               1.0, (float *)alphas, 1,
               (float *)X, numTrainingExamples, 0.0,
               (float *)W, 1 );
  }
  else
  {
    for( int i = 0; i < numTrainingExamples; i++ )
      alphas[i] *= y[i];
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
               1, numFeatures, numTrainingExamples,
               1.0, (double *)alphas, 1,
               (double *)X, numTrainingExamples, 0.0,
               (double *)W, 1 );
  }
  
  free(K);
  free(f);
  free(alphas);

} /* end svmTrain */

void calculateBI( floatType_t const *f, 
                  floatType_t const *alphas, 
                  floatType_t const *y,
                  int numTrainingExamples,
                  floatType_t *bLow, floatType_t *bHigh,
                  int *ILow, int *IHigh,
                  floatType_t const C )
{
  *bHigh = 100.0;
  *bLow = -100.0;
  for( int i = 0; i < numTrainingExamples; i++ )
  {  
    if( (floatType_t) 0.0 < alphas[i] && alphas[i] < C )
    {
/* set I_0 */
      MYMIN(*bHigh, f, i, *IHigh );
      MYMAX(*bLow, f, i, *ILow );
    } /* end if */
    else if( y[i] > (floatType_t) 0.0 )
    {
      if( alphas[i] == (floatType_t) 0.0 )
      {
/* set I_1 */
        MYMIN(*bHigh, f, i, *IHigh );
      } /* end if */
      else if( alphas[i] == C )
      {
/* set I_3 */
        MYMAX(*bLow, f, i, *ILow );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y > 0\n");
      } /* end if then else */ 
    } /* end if */
    else if( y[i] < (floatType_t) 0.0 )
    {
      if( alphas[i] == (floatType_t) 0.0 )
      {
/* set I_4 */
        MYMAX(*bLow, f, i, *ILow );
      } /* end if */
      else if( alphas[i] == C )
      {
/* set I_2 */
        MYMIN(*bHigh, f, i, *IHigh );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y < 0\n");
      } /* end if then else */ 
    } /* end if */
    else
    {
      printf("shouldn't be getting to this else \n");
    } /* end if */

  } /* end for */

} /* end calculateBI */     

void svmPredict( floatType_t const *X, 
                 floatType_t const *W, 
                 int const numExamples, int const numFeatures,
                 int *pred )
{
  floatType_t *p;

  p = (floatType_t *) malloc( sizeof(floatType_t) * numExamples );
  if( p == NULL ) fprintf(stderr,"error in malloc p in svmTrain\n");

//  for( int i = 0; i < numExamples; i++ ) p[i] = b;

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

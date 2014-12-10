#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers.h"


int main(int argc, char **argv) 
{

/* declare file pointers */

  char resultVectorFilename[]     = "y_vals.txt";
  char trainingSetFilename[]      = "X_vals.txt";
  char testSetFilename[]          = "testSet.txt";
  char testResultVectorFilename[] = "ytest.txt";

/* define constants */

  int featureVectorSize = FEATURE_VECTOR_SIZE;
  int trainingSize      = TRAINING_SET_SIZE;
  int testSize          = TEST_SET_SIZE;

/* define the arrays going to be used */

  int *resultVector, *trainingMatrix, *pred;
  floatType_t *X, *y, *K, *E, *alphas, *W;

/* declare various constants */

  int passes=0, maxPasses=5, numChangedAlphas, dots=12;
  floatType_t b=0.0, eta=0.0, L=0.0, H=0.0, tol=1.0e-3;
  floatType_t C=0.1;
  unsigned long seed = 8675309;

/* malloc resultVector */

  resultVector = (int *) malloc( sizeof(int) * trainingSize );
  if( resultVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

/* read resultVector from file */
 
  readMatrixFromFile( resultVectorFilename, resultVector, 
                      trainingSize, 1 );

/* malloc y */

  y = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( y == NULL ) 
    fprintf(stderr,"error malloc y\n");

/* copy result vector into y as float */

  for( int i = 0; i < trainingSize; i++ ) 
    y[i] = (floatType_t) resultVector[i];

/* malloc the training matrix.  each row is a different training
   example
*/

  trainingMatrix = (int *) malloc( sizeof(int) * trainingSize * 
                           featureVectorSize );
  if( trainingMatrix == NULL ) 
    fprintf(stderr,"Houston more problems\n");

/* read training examples from file as a matrix */

  readMatrixFromFile( trainingSetFilename, trainingMatrix, 
                      trainingSize, featureVectorSize );

/* mallox X */

  X = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize * 
                              featureVectorSize );
  if( X == NULL ) 
    fprintf(stderr,"error malloc X\n");

/* copy trainingMatrix into X as floats */

  for( int i = 0; i < trainingSize * featureVectorSize; i++ )
    X[i] = (floatType_t) trainingMatrix[i];

/* malloc K, the kernel matrix */

  K = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize * 
                              trainingSize );
  if( K == NULL ) 
    fprintf(stderr,"error malloc K\n");

/* malloc E */

  E = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( E == NULL ) 
    fprintf(stderr,"error malloc E\n");

/* zero out E */

  memset( E, 0, sizeof(floatType_t) * trainingSize );

/* malloc alphas */

  alphas = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( alphas == NULL ) fprintf(stderr,"error malloc alphas\n");

/* zero alphas */

  memset( alphas, 0, sizeof(floatType_t) * trainingSize );

/* map 0 values to -1 for training */

  for( int i = 0; i < trainingSize; i++ )
  {
    if( y[i] == 0.0 ) y[i] = -1.0;
  } /* end for */

/* compute the Kernel on every pair of examples.
   K = X * X'
*/

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans, 
               trainingSize, trainingSize, featureVectorSize,
               1.0, (float *)X, trainingSize, 
               (float *)X, trainingSize, 0.0, 
               (float *)K, trainingSize );
  }
  else
  {
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, 
               trainingSize, trainingSize, featureVectorSize,
               1.0, (double *)X, trainingSize, 
               (double *)X, trainingSize, 0.0, 
               (double *)K, trainingSize );
  }
  
               
  while( passes < maxPasses )
  {
    numChangedAlphas = 0;
    for( int i = 0; i < trainingSize; i++ )
    { 
      floatType_t tempSum = (floatType_t)0.0;
      for( int j = 0; j < trainingSize; j++ )
      {  
        tempSum += ( alphas[j] * y[j] * K[ INDX(j,i,trainingSize) ] );

      } /* end for j */

      E[i] = b + tempSum - y[i];

      if( (y[i]*E[i] < -tol && alphas[i] < C ) || 
           (y[i]*E[i] > tol  && alphas[i] > (floatType_t) 0.0 ) )
      {

        double rx = myRand( &seed );
        int j = floor( rx * double(trainingSize ) );

        tempSum = (floatType_t)0.0;
        for( int k = 0; k < trainingSize; k++ )
        {  
          tempSum += ( alphas[k] * y[k] * K[ INDX(k,j,trainingSize) ] );
        } /* end for j */
        
        E[j] = b + tempSum - y[j];

        floatType_t alphaIOld = alphas[i];
        floatType_t alphaJOld = alphas[j];


        if( y[i] == y[j] )
        {
          L = max( (floatType_t)0.0, alphas[j] + alphas[i] - C );
          H = min( C, alphas[j] + alphas[i] );
        } /* end if */
        else
        {
          L = max( (floatType_t)0.0, alphas[j] - alphas[i] );
          H = min( C, C + alphas[j] - alphas[i] );
        } /* end else */

        if( L == H ) continue;

        eta = (floatType_t)2.0 * K[INDX(i,j,trainingSize)] 
                   - K[INDX(i,i,trainingSize)] 
                   - K[INDX(j,j,trainingSize)];

        if( eta >= (floatType_t)0.0 ) continue;

        alphas[j] = alphas[j] - ( y[j] * ( E[i] - E[j] ) ) / eta;

        alphas[j] = min( H, alphas[j] );
        alphas[j] = max( L, alphas[j] );

        if( abs( alphas[j] - alphaJOld ) < tol )
        {
          alphas[j] = alphaJOld;
          continue;
        } /* end if */

        alphas[i] = alphas[i] + y[i] * y[j] * ( alphaJOld - alphas[j] );


        floatType_t b1 = b - E[i]
                     - y[i] * (alphas[i] - alphaIOld) * 
                            K[INDX(i,j,trainingSize)]
                     - y[j] * (alphas[j] - alphaJOld) * 
                            K[INDX(i,j,trainingSize)];

        floatType_t b2 = b - E[j]
                     - y[i] * (alphas[i] - alphaIOld) * 
                            K[INDX(i,j,trainingSize)]
                     - y[j] * (alphas[j] - alphaJOld) * 
                            K[INDX(j,j,trainingSize)];


        if( (floatType_t)0.0 < alphas[i] && alphas[i] < C ) b = b1;
        else if( (floatType_t)0.0 < alphas[j] && alphas[j] < C ) b = b2;
        else b = (b1 + b2) / (floatType_t)2.0;

        numChangedAlphas = numChangedAlphas + 1;

      } /* end if */
    } /* end for i */ 
   
    if( numChangedAlphas == 0 ) passes = passes + 1;
    else passes = 0; 

    fprintf(stdout,".");
    dots = dots + 1;
    if( dots > 78 )
    {
      dots = 0;
      fprintf(stdout,"\n");
    } 
    
   
  } /* end while */

  int *idx;
  idx = (int *) malloc( sizeof(int) * trainingSize );
  if( idx == NULL ) fprintf(stderr,"Houston we have a problem with IDX\n");

  for( int i = 0; i < trainingSize; i++ )
  {
    idx[i] = ( alphas[i] > 0.0f ) ? 1 : 0;
  } /* end for */


  W = (floatType_t *) malloc( sizeof(floatType_t) * featureVectorSize );
  if( W == NULL ) fprintf(stderr,"error malloc yW\n");

  if( sizeof( floatType_t ) == 4 )
  {
    for( int i = 0; i < trainingSize; i++ )
      alphas[i] *= y[i];
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 
               1, featureVectorSize, trainingSize,
               1.0, (float *)alphas, 1, 
               (float *)X, trainingSize, 0.0, 
               (float *)W, 1 );
  }
  else
  {
    for( int i = 0; i < trainingSize; i++ )
      alphas[i] *= y[i];
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 
               1, featureVectorSize, trainingSize,
               1.0, (double *)alphas, 1, 
               (double *)X, trainingSize, 0.0, 
               (double *)W, 1 );
  }

  pred = (int *) malloc( sizeof(int) * trainingSize );
  if( pred == NULL ) fprintf(stderr,"problem with malloc p in main\n");

  svmPredict( X, W, b, trainingSize, featureVectorSize, pred );
  
  double mean = 0.0;
  
  for( int i = 0; i < trainingSize; i++ ) 
  {
    mean += (pred[i] == resultVector[i]) ? 1.0 : 0.0;
  } /* end for */

  mean /= (double) trainingSize;
  printf("Prediction success rate on training set is %f\n",mean*100.0);


  readMatrixFromFile( testResultVectorFilename, resultVector, 
                      testSize, 1 );
  for( int i = 0; i < testSize; i++ ) 
    y[i] = (floatType_t) resultVector[i];

  readMatrixFromFile( testSetFilename, trainingMatrix, 
                      testSize, featureVectorSize );
  for( int i = 0; i < testSize * featureVectorSize; i++ )
    X[i] = (floatType_t) trainingMatrix[i];

  svmPredict( X, W, b, testSize, featureVectorSize, pred );

  mean = 0.0;
  
  for( int i = 0; i < testSize; i++ ) 
  {
    mean += (pred[i] == resultVector[i]) ? 1.0 : 0.0;
  } /* end for */

  mean /= (double) testSize;
  printf("Prediction success rate on test set is %f\n",mean*100.0);


  free(pred);
  free(W);
  free(E); 
  free(alphas);
  free(K);
  free(y);
  free(X);
  free( resultVector );
  free( trainingMatrix );
  return 0;
} /* end main */

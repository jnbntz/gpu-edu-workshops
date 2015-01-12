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


int main(int argc, char **argv) 
{

/* declare file pointers */

  char trainingVectorFilename[]   = "y_vals.txt";
  char trainingSetFilename[]      = "X_vals.txt";
  char testSetFilename[]          = "testSet.txt";
  char testResultVectorFilename[] = "ytest.txt";
  char sampleEmailFilename[]      = "emailVector.txt";

/* define constants */

  int const numFeatures         = FEATURE_VECTOR_SIZE;
  int const numTrainingExamples = TRAINING_SET_SIZE;
  int const numTestExamples     = TEST_SET_SIZE;
  floatType_t const tol         = 1.0e-3;
  floatType_t const C           = 0.1;
  int const maxPasses           = 5;
  char spam[]                   = "SPAM";
  char notSpam[]                = "NOT SPAM";

/* define the arrays going to be used */

  int *trainingVector, *trainingMatrix, *pred;
  int *testVector,     *testMatrix;
  floatType_t *X,   *Y,   *W, *Xtest;
  floatType_t *d_X, *d_Y, *d_W;

/* malloc trainingVector */

  trainingVector = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( trainingVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

/* read trainingVector from file */
 
  readMatrixFromFile( trainingVectorFilename, trainingVector, 
                      numTrainingExamples, 1 );

/* malloc y */

  Y = (floatType_t *) malloc( sizeof(floatType_t) * numTrainingExamples );
  if( Y == NULL ) 
    fprintf(stderr,"error malloc y\n");

/* copy result vector into y as float 
   aloso map 0 values to -1 for training */

  for( int i = 0; i < numTrainingExamples; i++ ) 
  {
    Y[i] = (floatType_t) trainingVector[i];
    if( Y[i] == 0.0 ) Y[i] = -1.0;
  } /* end for */

  CUDA_CALL( cudaMalloc( (void**)&d_Y, 
               sizeof(floatType_t) * numTrainingExamples ) );
  CUDA_CALL( cudaMemcpy( d_Y, Y, sizeof(floatType_t) * numTrainingExamples, 
               cudaMemcpyHostToDevice ) );

/* malloc the training matrix.  each row is a different training
   example
*/

  trainingMatrix = (int *) malloc( sizeof(int) * numTrainingExamples * 
                           numFeatures );
  if( trainingMatrix == NULL ) 
    fprintf(stderr,"Houston more problems\n");

/* read training examples from file as a matrix */

  readMatrixFromFile( trainingSetFilename, trainingMatrix, 
                      numTrainingExamples, numFeatures );

/* malloc X */

  X = (floatType_t *) malloc( sizeof(floatType_t) * numTrainingExamples * 
                              numFeatures );
  if( X == NULL ) 
    fprintf(stderr,"error malloc X\n");

/* copy trainingMatrix into X as floats */

  for( int i = 0; i < numTrainingExamples * numFeatures; i++ )
    X[i] = (floatType_t) trainingMatrix[i];

  CUDA_CALL( cudaMalloc( (void**) &d_X, 
               sizeof(floatType_t) * numFeatures * numTrainingExamples ) );
  CUDA_CALL( cudaMemcpy( d_X, X, 
               sizeof(floatType_t) * numFeatures * numTrainingExamples,  
               cudaMemcpyHostToDevice ) );

/* malloc the W matrix */

  W = (floatType_t *) malloc( sizeof(floatType_t) * numFeatures );
  if( W == NULL ) fprintf(stderr,"error malloc yW\n");

  CUDA_CALL( cudaMalloc( (void**) &d_W, sizeof(floatType_t) * numFeatures ) );
  CUDA_CALL( cudaMemset( d_W, 0, sizeof(floatType_t) * numFeatures ) );

/* setup timers */

  cudaEvent_t start, stop;
  CUDA_CALL( cudaEventCreate( &start ) );
  CUDA_CALL( cudaEventCreate( &stop ) );
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* call the training function */

  svmTrain(d_X, d_Y, C,
           numFeatures, numTrainingExamples,
           tol, maxPasses,
           d_W );

/* report time of svmTrain */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  float elapsedTime;
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time for svmTrain is %f sec\n",elapsedTime/1000.0f );

/* copy W matrix back to host for test of prediction */

  CUDA_CALL( cudaMemcpy( W, d_W, sizeof(floatType_t) * numFeatures,
               cudaMemcpyDeviceToHost ) );

/* malloc a prediction vector which will be the predicted values of the 
   results vector based on the training function 
*/

  pred = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( pred == NULL ) fprintf(stderr,"problem with malloc p in main\n");

/* start timer for svmTrain */

  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* call the predict function to populate the pred vector */

  svmPredict( X, W, numTrainingExamples, numFeatures, pred );

/* report time of svmTrain */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time for svmPredict is %f sec\n",elapsedTime/1000.0f );
  
/* calculate how well the predictions matched the actual values */

  double mean = 0.0;
  for( int i = 0; i < numTrainingExamples; i++ ) 
  {
    mean += (pred[i] == trainingVector[i]) ? 1.0 : 0.0;
  } /* end for */

  mean /= (double) numTrainingExamples;
  printf("Prediction success rate on training set is %f\n",mean*100.0);

/* malloc testVector */

  testVector = (int *) malloc( sizeof(int) * numTestExamples );
  if( testVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

/* read the test vector */

  readMatrixFromFile( testResultVectorFilename, testVector, 
                      numTestExamples, 1 );

/* malloc the test matrix.  each row is a different training
   example
*/

  testMatrix = (int *) malloc( sizeof(int) * numTestExamples * 
                           numFeatures );
  if( trainingMatrix == NULL ) 
    fprintf(stderr,"Houston more problems\n");

/* read the testSet data */

  readMatrixFromFile( testSetFilename, testMatrix, 
                      numTestExamples, numFeatures );

/* malloc Xtest */

  Xtest = (floatType_t *) malloc( sizeof(floatType_t) * numTestExamples * 
                              numFeatures );
  if( X == NULL ) 
    fprintf(stderr,"error malloc X\n");

/* copy the testMatrix into Xtest as floating point numbers */

  for( int i = 0; i < numTestExamples * numFeatures; i++ )
    Xtest[i] = (floatType_t) testMatrix[i];

/* predict the test set data using our original classifier */

  svmPredict( Xtest, W, numTestExamples, numFeatures, pred );

  mean = 0.0;
  for( int i = 0; i < numTestExamples; i++ ) 
  {
    mean += (pred[i] == testVector[i]) ? 1.0 : 0.0;
  } /* end for */

  mean /= (double) numTestExamples;
  printf("Prediction success rate on test set is %f\n",mean*100.0);

/* read the single test email data */

  readMatrixFromFile( sampleEmailFilename, testMatrix, 
                      1, numFeatures );

  for( int i = 0; i < numFeatures; i++ )
  {
    Xtest[i] = (floatType_t) testMatrix[i];
  }

/* predict whether the email is spam using our original classifier */

  svmPredict( Xtest, W, 1, numFeatures, pred );

  printf("Email test results 1 is SPAM 0 is NOT SPAM\n");
  printf("File Name %s, classification %d %s\n",
          sampleEmailFilename, pred[0], pred[0]==1 ? spam : notSpam);

  free(testVector);
  free(testMatrix);
  free(pred);
  free(W);
  free(Y);
  free(X);
  free(Xtest);
  free(trainingVector);
  free(trainingMatrix);

  CUDA_CALL( cudaFree( d_Y ) );
  CUDA_CALL( cudaFree( d_X ) );
  CUDA_CALL( cudaFree( d_W ) );
  
  cudaDeviceReset();

  return 0;
} /* end main */

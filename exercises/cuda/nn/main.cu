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

//  char trainingVectorFilename[]   = "y_vals.txt";
 // char trainingSetFilename[]      = "X_vals.txt";
  char trainingVectorFilename[]   = "TrainLabels.txt";
  char trainingSetFilename[]      = "Train.txt";
  char testSetFilename[]          = "testSet.txt";
  char testResultVectorFilename[] = "ytest.txt";
  char theta1Filename[]           = "Theta1.txt";
  char theta2Filename[]           = "Theta2.txt";

/* define constants */

  int const numFeatures         = FEATURE_VECTOR_SIZE;
  int const numTrainingExamples = TRAINING_SET_SIZE;
  int const numTestExamples     = TEST_SET_SIZE;
  int const sizeHiddenLayer     = HIDDEN_LAYER_SIZE;
  int const numClasses          = NUM_OUTPUT_CLASSES;
  floatType_t const tol         = 1.0e-3;
  floatType_t const C           = 0.1;
  char spam[]                   = "SPAM";
  char notSpam[]                = "NOT SPAM";
  floatType_t const eps         = 0.12;

/* define the arrays going to be used */

  float *trainingVector, *trainingMatrix, *pred;
  float *theta1, *theta2;
  float *theta1Grad, *theta2Grad;
  int *predictVector, *testMatrix;
  floatType_t *X, *Y, *W, *Xtest;

/* malloc trainingVector */

  trainingVector = (float *) malloc( sizeof(float) * numTrainingExamples );
  if( trainingVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

  memset( trainingVector, 0, sizeof(float)*numTrainingExamples );

/* read trainingVector from file */
 
  readMatrixFromFile( trainingVectorFilename, trainingVector, 
                      numTrainingExamples, 1, 1 );


/* malloc the training matrix.  each column is a different training
   example
*/

  trainingMatrix = (float *) malloc( sizeof(float) * numTrainingExamples * 
                           (numFeatures+1) );
  if( trainingMatrix == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( trainingMatrix, 0, sizeof(float)*
               numTrainingExamples*(numFeatures+1) );

/* read training examples from file as a matrix 
   read first column of data into second column of array to leave room for
   bias unit of ones
*/

//  readMatrixFromFile( trainingSetFilename, 
 //                     &trainingMatrix[INDX(0,1,numTrainingExamples)],
  //                    numTrainingExamples, numFeatures );
  readMatrixFromFile( trainingSetFilename, 
                      &trainingMatrix[1],
                      numFeatures, numTrainingExamples, numFeatures+1 );

  floatType_t scale = 1.0 / 256.0;
  for( int i = 0; i < (numFeatures+1)*numTrainingExamples; i++ )
    trainingMatrix[i] *= scale; 

/* malloc the theta1 matrix.  each row is a different training
   example
*/
  theta1 = (float *) malloc( sizeof(float) * sizeHiddenLayer * 
                           (numFeatures + 1 ) );
  if( theta1 == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta1, 0, sizeof(float)*sizeHiddenLayer*(numFeatures+1) );

/* read theta1 from file as a matrix */

//  readMatrixFromFile( theta1Filename, theta1,
 //                     sizeHiddenLayer, numFeatures+1 );
  for( int i = 0; i < sizeHiddenLayer*(numFeatures+1); i++ )
  {
    theta1[i] = double(rand()) / (double(RAND_MAX) + 1.0);
    theta1[i] *= (2.0*eps);
    theta1[i] -= eps;
//    printf("i %d theta2 %f\n",i,theta2[i]);
  } /* end for */

  theta1Grad = (float *) malloc( sizeof(float) * sizeHiddenLayer * 
                           (numFeatures + 1 ) );
  if( theta1Grad == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta1Grad, 0, sizeof(float)*sizeHiddenLayer*(numFeatures+1) );

/* malloc the theta2 matrix.  each row is a different training
   example
*/

  theta2 = (float *) malloc( sizeof(float) * numClasses * 
                           (sizeHiddenLayer + 1 ) );
  if( theta2 == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta2, 0, sizeof(float)*numClasses*(sizeHiddenLayer+1) );

/* read theta2 from file as a matrix */

//  readMatrixFromFile( theta2Filename, theta2,
 //                     numClasses, sizeHiddenLayer+1 );
  for( int i = 0; i < numClasses*(sizeHiddenLayer+1); i++ )
  {
    theta2[i] = double(rand()) / (double(RAND_MAX) + 1.0);
    theta2[i] *= (2.0*eps);
    theta2[i] -= eps;
//    printf("i %d theta2 %f\n",i,theta2[i]);
  } /* end for */

  theta2Grad = (float *) malloc( sizeof(float) * numClasses * 
                           (sizeHiddenLayer + 1 ) );
  if( theta2Grad == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta2Grad, 0, sizeof(float)*numClasses*(sizeHiddenLayer+1) );


  floatType_t cost;

#if 1
  trainNetwork( trainingMatrix, numTrainingExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                trainingVector );
#endif
#if 0
  costFunction( trainingMatrix, numTrainingExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                trainingVector, &cost, theta1Grad, theta2Grad );
#endif
 // printf("cost is %f\n",cost);

/* malloc testVector */

  predictVector = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( predictVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

  memset( predictVector, 0, sizeof(int)*numTrainingExamples );

  predict( trainingMatrix, numTrainingExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                predictVector );
  
  floatType_t result = 0.0;
  for( int i = 0; i < numTrainingExamples; i++ )
  {
//    printf("i %d actual %f predict %d\n",i,trainingVector[i],predictVector[i]); 
    if( (int) trainingVector[i] == predictVector[i] )
      result += (floatType_t) 1.0;
  } /* end for i */
  
  printf("total correct is %f\n",result);
  printf("prediction rate is %f\n",
      100.0 * result/(floatType_t)numTrainingExamples);

#if 0
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

/* malloc the Weight matrix */

  W = (floatType_t *) malloc( sizeof(floatType_t) * numFeatures );
  if( W == NULL ) fprintf(stderr,"error malloc yW\n");

/* setup timers */

  cudaEvent_t start, stop;
  CUDA_CALL( cudaEventCreate( &start ) );
  CUDA_CALL( cudaEventCreate( &stop ) );
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* call the training function */

  svmTrain(X, Y, C,
           numFeatures, numTrainingExamples,
           tol, W );

/* report time of svmTrain */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  float elapsedTime;
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time for svmTrain is %f sec\n",elapsedTime/1000.0f );

/* malloc a prediction vector which will be the predicted values of the 
   results vector based on the training function 
*/

  pred = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( pred == NULL ) fprintf(stderr,"problem with malloc p in main\n");

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
#endif
  return 0;
} /* end main */

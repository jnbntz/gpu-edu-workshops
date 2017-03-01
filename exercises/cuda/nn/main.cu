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

int main(int argc, char *argv[]) 
{
/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* declare file pointers */

  char trainingLabelFilename[]    = "train-labels.txt";
  char trainingSetFilename[]      = "train-images.txt";
  char testSetFilename[]          = "t10k-images.txt";
  char testLabelFilename[]        = "t10k-labels.txt";
#if 0
//used for debugging
  char theta1Filename[]           = "Theta1.txt";
  char theta2Filename[]           = "Theta2.txt";
#endif

/* define constants */

  int const numFeatures         = FEATURE_VECTOR_SIZE;
  int const numTrainingExamples = TRAINING_SET_SIZE;
  int const numTestExamples     = TEST_SET_SIZE;
  int const numClasses          = NUM_OUTPUT_CLASSES;
  floatType_t const eps         = 0.12;

/* define the arrays going to be used */

  float *trainingVector, *trainingMatrix;
  float *theta1, *theta2;
  float *testVector, *testMatrix;
  int *predictVector;

  float learningRate;
  int batchSize;
  int iterations;
  int sizeHiddenLayer;

/* read command line args if they're passed */

  readCommandLineArgs( argc, argv, &learningRate, &batchSize, &iterations, 
                       &sizeHiddenLayer );

  printf("Number of training examples           %d\n",numTrainingExamples);
  printf("Number of features/pixels per example %d\n",numFeatures);
  printf("Number of test examples               %d\n",numTestExamples);

/* malloc trainingVector, which are the labels of the trainin sets */

  trainingVector = (float *) malloc( sizeof(float) * numTrainingExamples );
  if( trainingVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

  memset( trainingVector, 0, sizeof(float)*numTrainingExamples );

/* read trainingVector from file */
 
  readMatrixFromFile( trainingLabelFilename, trainingVector, 
                      numTrainingExamples, 1, 1 );


/* malloc the training matrix.  each column is a different training
   example of 784 pixel values
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

/* scale the training matrix to 0 to 1, essentially a normalization 
   technique 
*/

  floatType_t scale = 1.0 / 256.0;
  for( int i = 0; i < (numFeatures+1)*numTrainingExamples; i++ )
    trainingMatrix[i] *= scale; 

/* malloc the theta1 matrix which are the weights for first layer
*/
  theta1 = (float *) malloc( sizeof(float) * sizeHiddenLayer * 
                           (numFeatures + 1 ) );
  if( theta1 == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta1, 0, sizeof(float)*sizeHiddenLayer*(numFeatures+1) );

/* init theta1 with random numbers */

  for( int i = 0; i < sizeHiddenLayer*(numFeatures+1); i++ )
  {
    theta1[i] = double(rand()) / (double(RAND_MAX) + 1.0);
    theta1[i] *= (2.0*eps);
    theta1[i] -= eps;
  } /* end for */

/* malloc the theta2 matrix which are weights for second layer
*/

  theta2 = (float *) malloc( sizeof(float) * numClasses * 
                           (sizeHiddenLayer + 1 ) );
  if( theta2 == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( theta2, 0, sizeof(float)*numClasses*(sizeHiddenLayer+1) );

/* init theta2 from random numbers */

  for( int i = 0; i < numClasses*(sizeHiddenLayer+1); i++ )
  {
    theta2[i] = double(rand()) / (double(RAND_MAX) + 1.0);
    theta2[i] *= (2.0*eps);
    theta2[i] -= eps;
  } /* end for */

/* setup timers using CUDA events */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );
#if 1
/* call the training function.  This is a majority of the runtime */

  trainNetwork( trainingMatrix, numTrainingExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                trainingVector, learningRate, iterations, batchSize );
#endif
/* report time of training */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time for training is            %.3e sec\n",
    elapsedTime/1000.0f );

/* malloc predictVector this is a vector that will be populated by the 
   predict function, i.e., it will take a set of pixel data and predict
   which digit it is, and put those values into a vector */

  predictVector = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( predictVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

  memset( predictVector, 0, sizeof(int)*numTrainingExamples );

/* test prediction on the training examples */

  predict( trainingMatrix, numTrainingExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                predictVector );

/* compare the predicted values versus the actual values, of the 
   training set 
*/
  
  floatType_t result = 0.0;
  for( int i = 0; i < numTrainingExamples; i++ )
  {
    if( (int) trainingVector[i] == predictVector[i] )
      result += (floatType_t) 1.0;
  } /* end for i */

  printf("Total correct on training set is      %d\n",(int)result);
  printf("Prediction rate of training set is    %.3f\n",
      100.0 * result/(floatType_t)numTrainingExamples);

/* malloc testVector. this is a test set of labels for data
   we haven't seen yet.
*/

  testVector = (float *) malloc( sizeof(float) * numTestExamples );
  if( testVector == NULL ) 
    fprintf(stderr,"Houston we have a problem\n");

  memset( testVector, 0, sizeof(float)*numTestExamples );

/* read testVector from file */
 
  readMatrixFromFile( testLabelFilename, testVector, 
                      numTestExamples, 1, 1 );

/* malloc the test matrix.  each column is a different test example of data
   we haven't seen before.
*/

  testMatrix = (float *) malloc( sizeof(float) * numTestExamples * 
                           (numFeatures+1) );
  if( testMatrix == NULL ) 
    fprintf(stderr,"Houston more problems\n");

  memset( testMatrix, 0, sizeof(float)*
               numTestExamples*(numFeatures+1) );

/* read test examples from file as a matrix 
   read first column of data into second column of array to leave room for
   bias unit of ones
*/

  readMatrixFromFile( testSetFilename, 
                      &testMatrix[1],
                      numFeatures, numTestExamples, numFeatures+1 );

/* scale the test matrix to 0 to 1 */

  scale = 1.0 / 256.0;
  for( int i = 0; i < (numFeatures+1)*numTestExamples; i++ )
    testMatrix[i] *= scale; 

  memset( predictVector, 0, sizeof(int)*numTestExamples );

/* test the prediction of test examples which we haven't trained on 
 */

  predict( testMatrix, numTestExamples, numFeatures+1,
                theta1, sizeHiddenLayer, numFeatures+1,
                theta2, numClasses, sizeHiddenLayer+1,
                predictVector );
  
  result = 0.0;
  for( int i = 0; i < numTestExamples; i++ )
  {
    if( (int) testVector[i] == predictVector[i] )
      result += (floatType_t) 1.0;
  } /* end for i */
  
  printf("Total correct on test set is          %d\n",(int)result);
  printf("Prediction rate of test set is        %.3f\n",
      100.0 * result/(floatType_t)numTestExamples);

  free(trainingVector);
  free(trainingMatrix);
  free(theta1);
  free(theta2);
  free(predictVector);
  free(testVector);
  free(testMatrix);

  return 0;
} /* end main */

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

/* declare variables with initial values */

  floatType_t b = 0.0;

/* define the arrays going to be used */

  int *trainingVector, *trainingMatrix, *pred;
  int *testVector, *testMatrix;
  floatType_t *X, *Y, *W, *Xtest;

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

/* copy result vector into y as float */

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

/* call the training function */

  svmTrain(X, Y, C,
           numFeatures, numTrainingExamples,
           tol, maxPasses,
           W, &b );

/* malloc a prediction vector which will be the predicted values of the 
   results vector based on the training function 
*/

  pred = (int *) malloc( sizeof(int) * numTrainingExamples );
  if( pred == NULL ) fprintf(stderr,"problem with malloc p in main\n");

/* call the predict function to populate the pred vector */

  svmPredict( X, W, b, numTrainingExamples, numFeatures, pred );
  
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

  svmPredict( Xtest, W, b, numTestExamples, numFeatures, pred );

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

  svmPredict( Xtest, W, b, 1, numFeatures, pred );

  printf("Email test results 1 is spam 0 is NOT spam\n");
  printf("File Name %s, classification %d\n",sampleEmailFilename, pred[0]);

  free(testVector);
  free(testMatrix);
  free(pred);
  free(W);
  free(Y);
  free(X);
  free(Xtest);
  free(trainingVector);
  free(trainingMatrix);
  return 0;
} /* end main */

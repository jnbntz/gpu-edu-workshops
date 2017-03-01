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

/* include the ATLAS headers */

extern "C"
{
#include <cblas.h>
}

/* choose precision to train and classify.  Only float and double are 
 * currently suppored
 */

typedef float floatType_t;

/* macro to convert 2d coords to 1d offset */

#define INDX(row,col,ld) (((col) * (ld)) + (row))

inline float sigmoid_f( float z )
{
  return 1.0f / ( 1.0f + expf( -z ) );
} /* end sigmoid_f */

inline double sigmoid( double z )
{
  return 1.0 / ( 1.0 + exp( -z ) );
} /* end sigmoid */

inline float sigmoidGradient_f( float z )
{
  float temp = sigmoid_f( z );
  return temp * ( 1.0f - temp );
} /* end sigGrad_f */

inline double sigmoidGradient( double z )
{
  double temp = sigmoid( z );
  return temp * ( 1.0 - temp );
} /* end sigGrad_f */


/* hardcoded constants for training and test set size and feature
 * vector size
 */

#define FEATURE_VECTOR_SIZE (784)
#define TRAINING_SET_SIZE (60000)
#define TEST_SET_SIZE (10000)
#define HIDDEN_LAYER_SIZE (25)
#define NUM_OUTPUT_CLASSES (10)

/* CUDA debugging */

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

/* function defs */

void readMatrixFromFile( char *, float *, const int, const int, const int );
void readCommandLineArgs( int, char *[], float *, int *, int *, int *);

void calcGradient( floatType_t *X,
                   int const XRows,
                   int const XCols,
                   floatType_t const *theta1,
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *theta2,
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y,
                   floatType_t *cost,
                   floatType_t       *theta1Grad,
                   floatType_t       *theta2Grad,
                   floatType_t       *tempMatrix );
void predict( floatType_t *X,
                   int const XRows,
                   int const XCols,
                   floatType_t const *theta1,
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *theta2,
                   int         const theta2Rows,
                   int         const theta2Cols,
                   int               *predictVector);
void trainNetwork( floatType_t       *X,
                   int         const XRows,
                   int         const XCols,
                   floatType_t       *theta1,
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t       *theta2,
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y,
                   float       const learningRate,
                   int         const iterations,
                   int         const batchSize );

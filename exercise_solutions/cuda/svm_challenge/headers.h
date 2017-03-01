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

#include "../debug.h"

/* choose precision to train and classify.  Only float and double are 
 * currently suppored
 */

typedef float floatType_t;

/* macro to convert 2d coords to 1d offset */

#define INDX(row,col,ld) (((col) * (ld)) + (row))

/* macros for max/min to combine with argmin */

#define MYMAX(val,array,i,index) \
if( array[i] > val ) \
{ \
  val = array[i]; \
  index = i; \
} \

#define MYMIN(val,array,i,index) \
if( array[i] < val ) \
{ \
  val = array[i]; \
  index = i; \
} \

/* macro to clip values from min to max */

#define CLIP(val,min,max) \
if( (val) < (min) ) val = (min); \
else if( (val) > (max) ) val = (max);

/* hardcoded constants for training and test set size and feature
 * vector size
 */

#define FEATURE_VECTOR_SIZE (1899)
#define TRAINING_SET_SIZE (4000)
#define TEST_SET_SIZE (1000)

/* function defs */

void readMatrixFromFile( char *, int *, const int, const int );

void svmTrain( floatType_t const *, floatType_t const *, floatType_t const,
               const int, const int,
               const floatType_t , const int,
               floatType_t * );

void svmPredict( floatType_t const *, floatType_t const *,  
                 int const, int const, int * );

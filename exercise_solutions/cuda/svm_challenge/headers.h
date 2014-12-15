
/* include the ATLAS headers */

extern "C"
{
#include <cblas.h>
}

typedef double floatType_t;

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

#define CLIP(val,min,max) \
if( (val) < (min) ) val = (min); \
else if( (val) > (max) ) val = (max);

#define FEATURE_VECTOR_SIZE (1899)
#define TRAINING_SET_SIZE (4000)
#define TEST_SET_SIZE (1000)

/* constants for the RNG */

#define AA (1664525UL)
#define CC (1013904223UL)
#define MM (4294967296UL)

/* function defs */

void readMatrixFromFile( char *, int *, const int, const int );

double myRand( unsigned long * );

void calculateBI( floatType_t const *,
                  floatType_t const *,
                  floatType_t const *,
                  int ,
                  floatType_t *, floatType_t *,
                  int *, int *,
                  floatType_t const );

void svmTrain( floatType_t const *, floatType_t const *, floatType_t const,
               const int, const int,
               const floatType_t , const int,
               floatType_t *, floatType_t * );

void svmPredict( floatType_t const *, floatType_t const *, floatType_t const, 
                 int const, int const, int * );

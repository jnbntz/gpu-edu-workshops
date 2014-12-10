
/* include the ATLAS headers */

extern "C"
{
#include <cblas.h>
}

typedef double floatType_t;

/* macro to convert 2d coords to 1d offset */

#define INDX(row,col,ld) (((col) * (ld)) + (row))

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
void svmPredict( floatType_t *, floatType_t *, floatType_t, 
                 int, int, int * );

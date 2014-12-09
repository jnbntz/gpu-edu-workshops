
extern "C"
{
#include <cblas.h>
}

typedef double floatType_t;

#define INDX(row,col,ld) (((col) * (ld)) + (row))

#define FEATURE_VECTOR_SIZE (1899)
#define TRAINING_SIZE (4000)
#define TEST_SIZE (1000)

#define AA (1664525UL)
#define CC (1013904223UL)
#define MM (4294967296UL)

void readMatrixFromFile( char *, int *, const int, const int );
void svmPredict( floatType_t *, floatType_t *, floatType_t, 
                 int, int, int * );

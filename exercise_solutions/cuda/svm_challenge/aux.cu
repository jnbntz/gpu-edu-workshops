#include <stdio.h>
#include "headers.h"

#define INDX(row,col,ld) (((col) * (ld)) + (row))

void svmPredict( floatType_t *X, floatType_t *W, floatType_t b, 
               const int numExamples, const int numFeatures,
               int *pred )
{

  floatType_t *p;

  p = (floatType_t *) malloc( sizeof(floatType_t) * numExamples );
  if( p == NULL ) fprintf(stderr,"error in malloc p in svmTrain\n");

  for( int i = 0; i < numExamples; i++ ) p[i] = b;

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemv( CblasColMajor, CblasNoTrans,
               numExamples, numFeatures,
               1.0, (float *)X, numExamples,
               (float *)W, 1, 1.0,
               (float *)p, 1 );
  }
  else
  {
    cblas_dgemv( CblasColMajor, CblasNoTrans,
               numExamples, numFeatures,
               1.0, (double *)X, numExamples,
               (double *)W, 1, 1.0,
               (double *)p, 1 );
  }

  for( int i = 0; i < numExamples; i++ )
    pred[i] = ( p[i] >= 0.0 ) ? 1 : 0;
 
  free(p);

} /* end svmTrain */


double myRand( unsigned long *seed )
{
    *seed = (AA * (*seed) + CC) % MM;
    double rx = (double)*seed / (double)MM; 
    return rx;
} /* end myRand */

void readMatrixFromFile( char *fileName, 
                          int *matrix, 
                     const int rows, 
                     const int cols )
{
  FILE *ifp;

  ifp = fopen( fileName, "r" );

  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening file %s\n", fileName);
    exit(911);
  } /* end if */

  for( int row = 0; row < rows; row++ )
  {
    for( int col = 0; col < cols; col++ )
    {
      if( !fscanf( ifp, "%d", 
          &matrix[ INDX( row, col, rows ) ] ) )
      {
        fprintf(stderr,"error reading training matrix file \n");
        exit(911);
      } /* end if */
    } /* end for col */
  } /* end for row */

  fclose(ifp);
  return;
} /* end readMatrixFromFile */

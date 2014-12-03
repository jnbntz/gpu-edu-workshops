#include <stdio.h>

#define INDX(row,col,ld) (((col) * (ld)) + (row))

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
//      printf("row %d col %d value %d\n",row,col,
 //       trainingMatrix[ INDX(row,col,trainingSize)]);
    } /* end for col */
  } /* end for row */

  fclose(ifp);
  return;
} /* end readMatrixFromFile */

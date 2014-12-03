#include <stdio.h>

#define INDX(row,col,ld) (((col) * (ld)) + (row))

#define FEATURE_VECTOR_SIZE 1899
#define TRAINING_SIZE 4000

void readMatrixFromFile( char *, int *, const int, const int );


int main(int argc, char **argv) 
{
  char resultVectorFilename[] = "y_vals.txt";
  char trainingSetFilename[] = "X_vals.txt";
  int featureVectorSize = FEATURE_VECTOR_SIZE;
  int trainingSize = TRAINING_SIZE;
  int *resultVector, *trainingMatrix;

  resultVector = (int *) malloc( sizeof(int) * trainingSize );
  if( resultVector == NULL ) fprintf(stderr,"Houston we have a problem\n");
 
  readMatrixFromFile( resultVectorFilename, resultVector, 
                      trainingSize, 1 );

  trainingMatrix = (int *) malloc( sizeof(int) * trainingSize * 
                           featureVectorSize );
  if( trainingMatrix == NULL ) fprintf(stderr,"Houston more problems\n");

  readMatrixFromFile( trainingSetFilename, trainingMatrix, 
                      trainingSize, featureVectorSize );

  for( int row = 0; row < trainingSize; row++ )
    printf("index %d resultVector %d\n",row, resultVector[row] );

  for( int row = 0; row < trainingSize; row++ )
  { 
    for( int col = 0; col < featureVectorSize; col++ )
      printf("row %d col %d value %d\n",row,col,trainingMatrix[INDX(row,col,trainingSize)]);
  } 

#if 0
  ifp = fopen( resultVectorFilename, "rt" );

  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening Results file\n");
    exit(911);
  } /* end if */

  int index = 0;

/* reading each element of the result vector */

  while( fgets( line, featureVectorSize, ifp ) != NULL )
  {
    printf("row is %d with length %d\n",index,strlen(line));
    if( sscanf( line, "%d", &resultVector[index] ) != 1 )
    {
      fprintf(stderr,"there was an issue reading the input file!\n");
    } /* end if */
    printf(" index %d resultVec %d\n",index,resultVector[index] );
    index++;
  } /* end while */
  
  printf("index is %d\n",index);

  fclose( ifp );

  trainingMatrix = (int *) malloc( sizeof(int) * trainingSize * 
                           featureVectorSize );
  if( trainingMatrix == NULL ) fprintf(stderr,"Houston more problems\n");

  ifp = fopen( trainingSetFilename, "r" );
  
  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening training set file\n");
    exit(911);
  } /* end if */

  for( int row = 0; row < trainingSize; row++ )
  {
    for( int col = 0; col < featureVectorSize; col++ )
    {
      if( !fscanf( ifp, "%d", 
          &trainingMatrix[ INDX( row, col, trainingSize ) ] ) )
      {
        fprintf(stderr,"error reading training matrix file \n");
        exit(911);
      } /* end if */

      printf("row %d col %d value %d\n",row,col,
        trainingMatrix[ INDX(row,col,trainingSize)]);
    } /* end for col */
  } /* end for row */
#endif
  return 0;
} /* end main */

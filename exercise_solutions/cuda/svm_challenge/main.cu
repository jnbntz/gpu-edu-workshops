#include <stdio.h>
#include <math.h>

extern "C" 
{
#include <cblas.h>
}

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
  int passes=0, maxPasses=5, numChangedAlphas;
  float *X, *y, *K, *E, *alphas;
  float b=0.0f, eta=0.0f, L=0.0f, H=0.0f, tol=1.0e-3;
  float C=0.1f;

  resultVector = (int *) malloc( sizeof(int) * trainingSize );
  if( resultVector == NULL ) fprintf(stderr,"Houston we have a problem\n");
 
  readMatrixFromFile( resultVectorFilename, resultVector, 
                      trainingSize, 1 );

  trainingMatrix = (int *) malloc( sizeof(int) * trainingSize * 
                           featureVectorSize );
  if( trainingMatrix == NULL ) fprintf(stderr,"Houston more problems\n");

  readMatrixFromFile( trainingSetFilename, trainingMatrix, 
                      trainingSize, featureVectorSize );

  y = (float *) malloc( sizeof(float) * trainingSize );
  if( y == NULL ) fprintf(stderr,"error malloc y\n");

  X = (float *) malloc( sizeof(float) * trainingSize * featureVectorSize );
  if( X == NULL ) fprintf(stderr,"error malloc X\n");

  K = (float *) malloc( sizeof(float) * trainingSize * trainingSize );
  if( K == NULL ) fprintf(stderr,"error malloc K\n");

  for( int i = 0; i < trainingSize; i++ ) 
    y[i] = (float) resultVector[i];

  for( int i = 0; i < trainingSize * featureVectorSize; i++ )
    X[i] = (float) trainingMatrix[i];

  E = (float *) malloc( sizeof(float) * trainingSize );
  if( E == NULL ) fprintf(stderr,"error malloc E\n");

  memset( E, 0, sizeof(float) * trainingSize );

  alphas = (float *) malloc( sizeof(float) * trainingSize );
  if( alphas == NULL ) fprintf(stderr,"error malloc alphas\n");

  memset( alphas, 0, sizeof(float) * trainingSize );

#if 0
  for( int row = 0; row < trainingSize; row++ )
    printf("index %d resultVector %d\n",row, resultVector[row] );

  for( int row = 0; row < trainingSize; row++ )
  { 
    for( int col = 0; col < featureVectorSize; col++ )
      printf("row %d col %d value %d\n",row,col,trainingMatrix[INDX(row,col,trainingSize)]);
  } 
#endif

/* map 0 values to -1 for training */

  for( int i = 0; i < trainingSize; i++ )
  {
    if( y[i] == 0.0f ) y[i] = -1.0f;
  } /* end for */
/* compute the Kernel on every pair of examples */

  cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans, 
               trainingSize, trainingSize, featureVectorSize,
               1.0f, X, trainingSize, 
               X, trainingSize, 0.0f, K, trainingSize );
               
#if 0
  for( int col = 0; col < trainingSize; col++ )
  { 
    for( int row = 0; row < trainingSize; row++ )
      printf("row %d col %d value %f\n",row,col,K[INDX(row,col,trainingSize)]);
    printf(" %d\n",(int) K[INDX(row,col,trainingSize)] );
  } 
#endif 

  while( passes < maxPasses )
  {
    numChangedAlphas = 0;
    for( int i = 0; i < trainingSize; i++ )
    { 
      float tempSum = 0.0f;

      for( int j = 0; j < trainingSize; j++ )
      {  
        tempSum += ( alphas[j] * y[j] * K[ INDX(j,i,trainingSize) ] );
      } /* end for j */

      E[i] = b + tempSum - y[i];

      if( (y[i]*E[i] < -tol && alphas[i] < C ) || 
           (y[i]*E[i] > tol  && alphas[i] > 0.0f ) )
      {

        int j = ceil( (float) trainingSize * 
                      float( rand() ) / ( float(RAND_MAX) + 1.0f ) );

        while( j == i ) 
          j = ceil( (float) trainingSize * 
                      float( rand() ) / ( float(RAND_MAX) + 1.0f ) );
        printf("j is %d\n",j);

        for( int k = 0; k < trainingSize; k++ )
        {  
          tempSum += ( alphas[k] * y[k] * K[ INDX(k,j,trainingSize) ] );
        } /* end for j */
        
        E[j] = b + tempSum - y[j];

        float alphaIOld = alphas[i];
        float alphaJOld = alphas[j];

        if( y[i] == y[j] )
        {
          L = max( 0.0f, alphas[j] + alphas[i] - C );
          H = min( C, alphas[j] + alphas[i] );
        } /* end if */
        else
        {
          L = max( 0.0f, alphas[j] - alphas[i] );
          H = min( C, C + alphas[j] - alphas[i] );
        } /* end else */

        if( L == H ) continue;

        eta = 2.0f * K[INDX(i,j,trainingSize)] 
                   - K[INDX(i,i,trainingSize)] 
                   - K[INDX(j,j,trainingSize)];

        if( eta >= 0.0f ) continue;

        alphas[j] = alphas[j] - ( y[j] * ( E[i] - E[j] ) ) / eta;

        alphas[j] = min( H, alphas[j] );
        alphas[j] = max( L, alphas[j] );

        if( abs( alphas[j] - alphaJOld ) < tol )
        {
          alphas[j] = alphaJOld;
          continue;
        } /* end if */

        float b1 = b - E[i]
                     - y[i] * (alphas[i] - alphaIOld) * 
                            K[INDX(i,j,trainingSize)]
                     - y[j] * (alphas[j] - alphaJOld) * 
                            K[INDX(i,j,trainingSize)];


      } /* end if */

      exit(911);
    } /* end for i */ 
  } /* end while */

  free(E); 
  free(alphas);
  free(K);
  free(y);
  free(X);
  free( resultVector );
  free( trainingMatrix );
  return 0;
} /* end main */

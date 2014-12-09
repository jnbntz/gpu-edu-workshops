#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers.h"


int main(int argc, char **argv) 
{
  char resultVectorFilename[] = "y_vals.txt";
  char trainingSetFilename[] = "X_vals.txt";
  int featureVectorSize = FEATURE_VECTOR_SIZE;
  int trainingSize = TRAINING_SIZE;
  int *resultVector, *trainingMatrix;
  int passes=0, maxPasses=5, numChangedAlphas, dots=12, *pred;
  floatType_t *X, *y, *K, *E, *alphas, *W;
  floatType_t b=0.0, eta=0.0, L=0.0, H=0.0, tol=1.0e-3;
  floatType_t C=0.1;

unsigned long seed = 8675309;

  resultVector = (int *) malloc( sizeof(int) * trainingSize );
  if( resultVector == NULL ) fprintf(stderr,"Houston we have a problem\n");
 
  readMatrixFromFile( resultVectorFilename, resultVector, 
                      trainingSize, 1 );

  trainingMatrix = (int *) malloc( sizeof(int) * trainingSize * 
                           featureVectorSize );
  if( trainingMatrix == NULL ) fprintf(stderr,"Houston more problems\n");

  readMatrixFromFile( trainingSetFilename, trainingMatrix, 
                      trainingSize, featureVectorSize );

  y = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( y == NULL ) fprintf(stderr,"error malloc y\n");

  X = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize * featureVectorSize );
  if( X == NULL ) fprintf(stderr,"error malloc X\n");

  K = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize * trainingSize );
  if( K == NULL ) fprintf(stderr,"error malloc K\n");

  for( int i = 0; i < trainingSize; i++ ) 
    y[i] = (floatType_t) resultVector[i];

  for( int i = 0; i < trainingSize * featureVectorSize; i++ )
    X[i] = (floatType_t) trainingMatrix[i];

  E = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( E == NULL ) fprintf(stderr,"error malloc E\n");

  memset( E, 0, sizeof(floatType_t) * trainingSize );

  alphas = (floatType_t *) malloc( sizeof(floatType_t) * trainingSize );
  if( alphas == NULL ) fprintf(stderr,"error malloc alphas\n");

  memset( alphas, 0, sizeof(floatType_t) * trainingSize );

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
    if( y[i] == 0.0 ) y[i] = -1.0;
  } /* end for */
/* compute the Kernel on every pair of examples */

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans, 
               trainingSize, trainingSize, featureVectorSize,
               1.0, (float *)X, trainingSize, 
               (float *)X, trainingSize, 0.0, 
               (float *)K, trainingSize );
  }
  else
  {
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, 
               trainingSize, trainingSize, featureVectorSize,
               1.0, (double *)X, trainingSize, 
               (double *)X, trainingSize, 0.0, 
               (double *)K, trainingSize );
  }
  
               
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
      floatType_t tempSum = (floatType_t)0.0;
      for( int j = 0; j < trainingSize; j++ )
      {  
        tempSum += ( alphas[j] * y[j] * K[ INDX(j,i,trainingSize) ] );

      } /* end for j */

      E[i] = b + tempSum - y[i];

      if( (y[i]*E[i] < -tol && alphas[i] < C ) || 
           (y[i]*E[i] > tol  && alphas[i] > (floatType_t) 0.0 ) )
      {

        int j;
#if 0
        int j = ceil( (float) trainingSize * 
                      float( rand() ) / ( float(RAND_MAX) + 1.0f ) );

        while( j == i ) 
          j = ceil( (float) trainingSize * 
                      float( rand() ) / ( float(RAND_MAX) + 1.0f ) );
#endif
//        printf("seed before %ld\n",seed);
 //       printf("ans = %ld\n",AA * seed + CC);
        seed = (AA * seed + CC) % MM; 
//        printf("AA %ld\n",AA);
 //       printf("CC %ld\n",CC);
  //      printf("MM %ld\n",MM);
   //     printf("seed = %ld\n",seed);
       double rx = (double)seed / (double)MM;
   //     printf("rx = %f\n",rx);
//        j = 3918;
      j = floor( rx * double(trainingSize ) );
  //     j = (i + 1) % trainingSize;
//       printf("j =  %d\n",j+1);
//      if( j > 3990 ) exit(911);

        tempSum = (floatType_t)0.0;
        for( int k = 0; k < trainingSize; k++ )
        {  
          tempSum += ( alphas[k] * y[k] * K[ INDX(k,j,trainingSize) ] );
//        if( alphas[k] != 0.0f ) 
 //          printf("k %d alphas %f y %f K %f\n",k+1,alphas[k],
  //           y[k],K[INDX(k,j,trainingSize)] );
        } /* end for j */
        
        E[j] = b + tempSum - y[j];

//printf("b = %f\n",b);
//printf("tempsum = %f\n",tempSum);
//printf("Yj = %f\n",y[j] );
//printf("E_j = %f\n",E[j]);

        floatType_t alphaIOld = alphas[i];
        floatType_t alphaJOld = alphas[j];

//        printf("alphaIOld %f alphaJOld %f\n",alphaIOld,alphaJOld);
 //       printf("yi %f yj %f\n",y[i],y[j]);

        if( y[i] == y[j] )
        {
          L = max( (floatType_t)0.0, alphas[j] + alphas[i] - C );
          H = min( C, alphas[j] + alphas[i] );
        } /* end if */
        else
        {
          L = max( (floatType_t)0.0, alphas[j] - alphas[i] );
          H = min( C, C + alphas[j] - alphas[i] );
        } /* end else */

  //      printf("L %f H %f\n",L,H);

        if( L == H ) continue;

        eta = (floatType_t)2.0 * K[INDX(i,j,trainingSize)] 
                   - K[INDX(i,i,trainingSize)] 
                   - K[INDX(j,j,trainingSize)];

//        printf("eta %f\n",eta);

        if( eta >= (floatType_t)0.0 ) continue;

        alphas[j] = alphas[j] - ( y[j] * ( E[i] - E[j] ) ) / eta;

        alphas[j] = min( H, alphas[j] );
        alphas[j] = max( L, alphas[j] );

        if( abs( alphas[j] - alphaJOld ) < tol )
        {
          alphas[j] = alphaJOld;
          continue;
        } /* end if */

        alphas[i] = alphas[i] + y[i] * y[j] * ( alphaJOld - alphas[j] );


        floatType_t b1 = b - E[i]
                     - y[i] * (alphas[i] - alphaIOld) * 
                            K[INDX(i,j,trainingSize)]
                     - y[j] * (alphas[j] - alphaJOld) * 
                            K[INDX(i,j,trainingSize)];

        floatType_t b2 = b - E[j]
                     - y[i] * (alphas[i] - alphaIOld) * 
                            K[INDX(i,j,trainingSize)]
                     - y[j] * (alphas[j] - alphaJOld) * 
                            K[INDX(j,j,trainingSize)];

 //       printf("b1 = %f\n",b1);
  //      printf("b2 = %f\n",b2);
   //     printf("alphas(i) = %f\n",alphas[i]);
    //    printf("alphas(j) = %f\n",alphas[j]);


        if( (floatType_t)0.0 < alphas[i] && alphas[i] < C ) b = b1;
        else if( (floatType_t)0.0 < alphas[j] && alphas[j] < C ) b = b2;
        else b = (b1 + b2) / (floatType_t)2.0;

//printf("b is %f\n",b);

        numChangedAlphas = numChangedAlphas + 1;

//        printf("numChangedAlphas %d\n",numChangedAlphas );

      } /* end if */

    } /* end for i */ 
   
    if( numChangedAlphas == 0 ) passes = passes + 1;
    else passes = 0; 

//printf("new pass\n\n");
 //   printf("b = %f\n",b);

    fprintf(stdout,".");
    dots = dots + 1;
//    if( dots == 14 ) exit(911);
    if( dots > 78 )
    {
      dots = 0;
      fprintf(stdout,"\n");
    } 
    
   
  } /* end while */

  int *idx;
  idx = (int *) malloc( sizeof(int) * trainingSize );
  if( idx == NULL ) fprintf(stderr,"Houston we have a problem with IDX\n");

  for( int i = 0; i < trainingSize; i++ )
  {
    idx[i] = ( alphas[i] > 0.0f ) ? 1 : 0;
//    printf(" %d\n",idx[i] );
  } /* end for */

//  printf("b is %f\n",b);

  W = (floatType_t *) malloc( sizeof(floatType_t) * featureVectorSize );
  if( W == NULL ) fprintf(stderr,"error malloc yW\n");

  if( sizeof( floatType_t ) == 4 )
  {
    for( int i = 0; i < trainingSize; i++ )
      alphas[i] *= y[i];
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 
               1, featureVectorSize, trainingSize,
               1.0, (float *)alphas, 1, 
               (float *)X, trainingSize, 0.0, 
               (float *)W, 1 );
  }
  else
  {
    for( int i = 0; i < trainingSize; i++ )
      alphas[i] *= y[i];
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 
               1, featureVectorSize, trainingSize,
               1.0, (double *)alphas, 1, 
               (double *)X, trainingSize, 0.0, 
               (double *)W, 1 );
  }

//  for( int i = 0; i < featureVectorSize; i++ )
 //   printf("%f\n",W[i]);

//  p = alphas;
  pred = (int *) malloc( sizeof(int) * trainingSize );
  if( pred == NULL ) fprintf(stderr,"problem with malloc p in main\n");

#if 0
  for( int i = 0; i < trainingSize; i++ ) p[i] = b;

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemv( CblasColMajor, CblasNoTrans,
               trainingSize, featureVectorSize,
               1.0, (float *)X, trainingSize,
               (float *)W, 1, 1.0,
               (float *)p, 1 );
  }
  else
  {
    cblas_dgemv( CblasColMajor, CblasNoTrans,
               trainingSize, featureVectorSize,
               1.0, (double *)X, trainingSize,
               (double *)W, 1, 1.0,
               (double *)p, 1 );
  }
#endif

  svmPredict( X, W, b, trainingSize, featureVectorSize, pred );
  
  double mean = 0.0;
  
  for( int i = 0; i < trainingSize; i++ ) 
  {
//    int prediction = (p[i] >= 0.0) ? 1 : 0;
    mean += (pred[i] == resultVector[i]) ? 1.0 : 0.0;
  } /* end for */
  mean /= (double) trainingSize;
  printf("Prediction success rate is %f\n",mean*100.0);

  free(pred);
  free(W);
  free(E); 
  free(alphas);
  free(K);
  free(y);
  free(X);
  free( resultVector );
  free( trainingMatrix );
  return 0;
} /* end main */

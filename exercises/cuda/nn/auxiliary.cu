/*
 *  Copyright 2014 NVIDIA Corporation
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers.h"

void costFunction( floatType_t *X, 
                   int const XRows, 
                   int const XCols,
                   floatType_t const *theta1, 
                   int         const theta1Rows,
                   int         const theta1Cols,
                   floatType_t const *theta2, 
                   int         const theta2Rows,
                   int         const theta2Cols,
                   floatType_t const *Y, 
                   floatType_t *cost )
{

  floatType_t *tempMatrix, *z2, *a2, *a3;

printf("Xrows %d Xcols %d\n",XRows,XCols);
printf("t1row %d t1col %d\n",theta1Rows,theta1Cols);
printf("t2row %d t2col %d\n",theta2Rows,theta2Cols);

  for( int i = 0; i < XRows; i++ ) X[i] = (floatType_t) 1.0;
  
  tempMatrix = (floatType_t *) malloc( sizeof(floatType_t) *
                               ( XRows * (theta1Rows+1) + 
                                 XRows * (theta1Rows+1) +
                                 XRows * (theta2Rows+1) ) );

  z2 = tempMatrix;
  a2 = &z2[INDX(XRows,theta1Rows,XRows)];
  a3 = &a2[INDX(XRows,theta1Rows+1,XRows)];

  if( sizeof( floatType_t ) == 4 ) 
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 XRows, theta1Rows, theta1Cols,
                 1.0f, (float *) X, XRows,
                 (float *) theta1, theta1Rows, 0.0f,
                 (float *) &z2[INDX(0,1,XRows)], XRows );
//                 (float *) &tempMatrix[INDX(0,1,XRows)], XRows );
    for( int j = 1; j < theta1Rows+1; j++ )
      for( int i = 0; i < XRows; i++ )
        a2[INDX(i,j,XRows)] = 
          sigmoid_f( z2[INDX(i,j,XRows)] );
  } /* end if */
  else
  {
  } /* end else */  



  for( int i = 0; i < XRows; i++ ) 
    a2[INDX(i,0,XRows)] = (floatType_t) 1.0;

//  a3 = &tempMatrix[INDX(0,theta2Cols+1,XRows)];

  if( sizeof( floatType_t ) == 4 )
  {
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                 XRows, theta2Rows, theta2Cols,
                 1.0f, (float *) a2, XRows,
                 (float *) theta2, theta2Rows, 0.0f,
                 (float *) a3, XRows );
//                 (float *) &tempMatrix[INDX(0,theta2Cols+1,XRows)], XRows );
    for( int j = 0; j < theta2Rows; j++ )
      for( int i = 0; i < XRows; i++ )
        a3[INDX(i,j,XRows)] = 
          sigmoid_f( a3[INDX(i,j,XRows)] );
  } /* end if */
  else
  { 
  } /* end else */

//  for( int i = 0; i < theta2Rows; i++ )
 //   printf("col %d val %f\n",i,a3[INDX(4999,i,XRows)] );
//    printf("col %d val %f\n",i,tempMatrix[INDX(4999,theta2Cols+1+i,XRows)] );

  


//  for( int i = 0; i < theta2Rows; i++ )
 //   printf("col %d val %e\n",i,a3[INDX(4999,i,XRows)] );

  floatType_t yTemp[11];
  floatType_t jTemp = 0.0;

  for( int row = 0; row < XRows; row++ )
  {
    memset( yTemp, 0, sizeof(floatType_t) * 11 ); 
//   printf("row %d Y %f %d\n",row,Y[row],(int) Y[row] );
    yTemp[  (int)Y[row]  ] = (floatType_t) 1.0;
    for( int j = 1; j <= theta2Rows; j++ )
    {
//      printf("j %d val %f\n",j,-log(a3[INDX(row,j-1,XRows)])*yTemp[j]);
      jTemp += -log( a3[INDX(row,j-1,XRows)] ) * yTemp[j] 
             - ( log( (floatType_t) 1.0 - a3[INDX(row,j-1,XRows)] ) * 
                 ( (floatType_t) 1.0 - yTemp[j] ) ) ;
    } /* end for */
//    printf("row=%d jTemp is %f\n",row,jTemp);
  } /* end for */

  jTemp /= (floatType_t) XRows;
  printf("jTemp is %f %f\n",jTemp, jTemp / (floatType_t)XRows );

  floatType_t *tempY, *delta3;
  tempY = (floatType_t *)malloc( sizeof(floatType_t)*11);
  delta3 = tempY;

  floatType_t *delta2;
  delta2 = (floatType_t *)malloc( sizeof(floatType_t) * theta2Cols );

  for( int row = 0; row < XRows; row++ )
  { 
    memset( tempY, 0, sizeof( floatType_t) * 11 );
    tempY[ (int) Y[row] ] = (floatType_t) 1.0;
    for( int j = 0; j < 10; j++ ) 
    {
      tempY[j+1] = a3[INDX(row,j,XRows)] - tempY[j+1];
//      printf("j %d delta3 %e\n",j+1, tempY[j+1] );
    } /* end for j */

    if( sizeof( floatType_t ) == 4 )
    {
      cblas_sgemv( CblasColMajor, CblasTrans,
                 theta2Rows, theta2Cols,
                 1.0f, theta2, theta2Rows, 
                 &delta3[1], 1, 0.0f,
                 delta2, 1 );

      for( int j = 1; j <= theta1Rows; j++ )
      {
        delta2[j] *= sigmoidGradient_f( z2[INDX(row,j,XRows)] );
      } /* end for */
    } /* end if */
    else
    { 
    } /* end else */



//    for( int j = 0; j <= theta1Rows; j++ )
 //    printf("j %d val %e\n",j,delta2[j] );
  //  printf("\n");

  } /* end for row */



  free(tempMatrix);
  free(tempY);
  free(delta2);
} /* end costFunction */

void readMatrixFromFile( char *fileName, 
                         float *matrix, 
                         int const rows, 
                         int const cols )
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
      if( !fscanf( ifp, "%f", 
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

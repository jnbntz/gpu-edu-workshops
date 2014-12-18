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

__global__ void k_updateF( floatType_t *f, floatType_t *alphas,
                         int         const IHigh,
                         int         const ILow,
                         floatType_t const *K,
                         int         const numTrainingExamples,
                         floatType_t const *y,
                         floatType_t const C,
                         floatType_t const bLow, 
                         floatType_t const bHigh )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  for( int i = 0; i < numTrainingExamples; i++ )
  if( idx < numTrainingExamples )
  {
#if 1
/* grab alpha values */

    floatType_t ail  = alphas[ILow];
    floatType_t aih  = alphas[IHigh];

/* calculate eta */

    floatType_t eta = K[INDX(IHigh,IHigh,numTrainingExamples)]
        + K[INDX(ILow, ILow, numTrainingExamples)]
        - (floatType_t)2.0 * K[INDX(IHigh,ILow,numTrainingExamples)];

/* calculate new alpha values */

    floatType_t ailp  = ail + ( y[ILow] * ( bHigh - bLow ) ) / eta;
    floatType_t aihp  = aih +
                    y[ILow] * y[IHigh] * ( ail - ailp );

#endif
//    floatType_t ailp = alphaILowPrime;
 //   floatType_t aihp = alphaIHighPrime;
  //  floatType_t ail  = alphaILow;
   // floatType_t aih = alphaIHigh;
    CLIP( ailp, (floatType_t) 0.0, C );
    CLIP( aihp, (floatType_t) 0.0, C );

/* update alpha values in the array */
#if 0
    if( idx == 0 )
    {
      alphas[ILow]  = alphaILowPrime;
      alphas[IHigh] = alphaIHighPrime;
    }
#else
    if( idx == 0 )
    {
      alphas[ILow]  = ailp;
      alphas[IHigh] = aihp;
    }
#endif
#if 0
    f[idx] = f[idx]
         + ( ( alphaIHighPrime - alphaIHigh )
             * y[IHigh] * K[INDX(IHigh,idx,numTrainingExamples)] )
         + ( ( alphaILowPrime - alphaILow )
             * y[ILow] * K[INDX(ILow,idx,numTrainingExamples)] );
#else
    f[idx] = f[idx]
         + ( ( aihp - aih )
             * y[IHigh] * K[INDX(IHigh,idx,numTrainingExamples)] )
         + ( ( ailp - ail )
             * y[ILow] * K[INDX(ILow,idx,numTrainingExamples)] );
#endif
  } /* end for i */
  return; 
} /* end updateF */ 

__global__ void k_calculateBI( floatType_t const *f,
                  floatType_t const *alphas,
                  floatType_t const *y,
                  int numTrainingExamples,
                  floatType_t *bLow, floatType_t *bHigh,
                  int *ILow, int *IHigh,
                  floatType_t const C )
{
  __shared__ floatType_t bHighAr[1024];
  __shared__ floatType_t bLowAr[1024];
  __shared__ int         IHighAr[1024];
  __shared__ int         ILowAr[1024];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  bHighAr[threadIdx.x] = 100.0;
  bLowAr[threadIdx.x]  = -100.0;
  IHighAr[threadIdx.x] = -999;
  ILowAr[threadIdx.x] = -999;

  while( idx < numTrainingExamples )
  {
//  for( int idx = 0; idx < numTrainingExamples; idx++ )
 // {
    if( (floatType_t) 0.0 < alphas[idx] && alphas[idx] < C )
    {
/* set I_0 */
      MYMIN(bHighAr[threadIdx.x], f, idx, IHighAr[threadIdx.x] );
      MYMAX(bLowAr[threadIdx.x], f, idx, ILowAr[threadIdx.x] );
    } /* end if */
    else if( y[idx] > (floatType_t) 0.0 )
    {
      if( alphas[idx] == (floatType_t) 0.0 )
      {
/* set I_1 */
        MYMIN(bHighAr[threadIdx.x], f, idx, IHighAr[threadIdx.x] );
      } /* end if */
      else if( alphas[idx] == C )
      {
/* set I_3 */
        MYMAX(bLowAr[threadIdx.x], f, idx, ILowAr[threadIdx.x] );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y > 0\n");
      } /* end if then else */
    } /* end if */
    else if( y[idx] < (floatType_t) 0.0 )
    {
      if( alphas[idx] == (floatType_t) 0.0 )
      {
/* set I_4 */
        MYMAX(bLowAr[threadIdx.x], f, idx, ILowAr[threadIdx.x] );
      } /* end if */
      else if( alphas[idx] == C )
      {
/* set I_2 */
        MYMIN(bHighAr[threadIdx.x], f, idx, IHighAr[threadIdx.x] );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y < 0\n");
      } /* end if then else */
    } /* end if */
    else
    {
      printf("shouldn't be getting to this else \n");
    } /* end else */

     

//  } /* end for */
    idx += blockDim.x;
  } /* end while */
#if 1
  if( threadIdx.x == 0 )
  { 
    for( int i = 1; i < blockDim.x; i++ )
    {
      if( bHighAr[i] < bHighAr[0] )
      {  
        bHighAr[0] = bHighAr[i];
        IHighAr[0] = IHighAr[i];
      } /* end if */
      if( bLowAr[i] > bLowAr[0] )
      {
        bLowAr[0] = bLowAr[i];
        ILowAr[0] = ILowAr[i];
      } /* end if */
    } /* end for */

    *bHigh = bHighAr[0];
    *bLow  = bLowAr[0];
    *IHigh = IHighAr[0];
    *ILow  = ILowAr[0]; 
  } /* end if */
#endif

  
  return;
} /* end calculateBI */

__global__ void k_scaleAlpha( floatType_t *alphas,
                              floatType_t const *y,
                              int const numTrainingExamples )
{

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if( idx < numTrainingExamples )
    alphas[idx] *= y[idx];
  return;
} /* end k_scaleAlpha */

__global__ void k_initF( floatType_t *f,
                         floatType_t const *y,
                         int const numTrainingExamples )
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if( idx < numTrainingExamples )
    f[idx] = -y[idx];
  return; 
} /* end k_initF */

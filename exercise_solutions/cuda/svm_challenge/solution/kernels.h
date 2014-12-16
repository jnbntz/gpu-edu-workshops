

__global__ void k_updateF( floatType_t *f, 
                         floatType_t const alphaIHighPrime,
                         floatType_t const alphaIHigh,
                         int         const IHigh,
                         floatType_t const alphaILowPrime,
                         floatType_t const alphaILow,
                         int         const ILow,
                         floatType_t const *K,
                         int         const numTrainingExamples,
                         floatType_t const *y )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  for( int i = 0; i < numTrainingExamples; i++ )
  if( idx < numTrainingExamples )
  {
    f[idx] = f[idx]
         + ( ( alphaIHighPrime - alphaIHigh )
             * y[IHigh] * K[INDX(IHigh,idx,numTrainingExamples)] )
         + ( ( alphaILowPrime - alphaILow )
             * y[ILow] * K[INDX(ILow,idx,numTrainingExamples)] );
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
  *bHigh = 100.0;
  *bLow = -100.0;
  for( int i = 0; i < numTrainingExamples; i++ )
  {
    if( (floatType_t) 0.0 < alphas[i] && alphas[i] < C )
    {
/* set I_0 */
      MYMIN(*bHigh, f, i, *IHigh );
      MYMAX(*bLow, f, i, *ILow );
    } /* end if */
    else if( y[i] > (floatType_t) 0.0 )
    {
      if( alphas[i] == (floatType_t) 0.0 )
      {
/* set I_1 */
        MYMIN(*bHigh, f, i, *IHigh );
      } /* end if */
      else if( alphas[i] == C )
      {
/* set I_3 */
        MYMAX(*bLow, f, i, *ILow );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y > 0\n");
      } /* end if then else */
    } /* end if */
    else if( y[i] < (floatType_t) 0.0 )
    {
      if( alphas[i] == (floatType_t) 0.0 )
      {
/* set I_4 */
        MYMAX(*bLow, f, i, *ILow );
      } /* end if */
      else if( alphas[i] == C )
      {
/* set I_2 */
        MYMIN(*bHigh, f, i, *IHigh );
      } /* end if */
      else
      {
        printf("Shouldn't be getting to this else! y < 0\n");
      } /* end if then else */
    } /* end if */
    else
    {
      printf("shouldn't be getting to this else \n");
    } /* end if */

  } /* end for */
  return;
} /* end calculateBI */


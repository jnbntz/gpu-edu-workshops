#include <stdio.h>

int main(int argc, char **argv) 
{

  FILE *ifp;
  char *mode = "r";
  char inputFilename[] = "vocab.txt";
  
  ifp = fopen( inputFilename, "r" );

  if( ifp == NULL ) 
  {
    fprintf(stderr, "Error opening input file\n");
    exit(911);
  } /* end if */

  return 0;
} /* end main */

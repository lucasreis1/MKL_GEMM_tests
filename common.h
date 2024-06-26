#include "mkl.h"

#ifdef __cplusplus
extern "C" {
#endif

float b2f(MKL_BF16 src); 
MKL_BF16 f2b(float src);
MKL_F16 f2h(float x); 
float h2f(MKL_F16 x);

void printMatrix(float *M, int m, int n);

float * readFromFile(const char *fileName, int *size);

#ifdef __cplusplus
}
#endif

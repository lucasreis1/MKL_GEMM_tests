#include "common.h"
#include <chrono>
#include <cstdio>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MKL_INT         m, n, k;
  MKL_INT         lda, ldb, ldc;
  MKL_INT         rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
  double alpha, beta;
  double *a, *b, *c;
  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE transA, transB;
  MKL_INT ma, na, mb, nb;

  transA = transB = CblasNoTrans;

  /*       Get input parameters                                  */

  if (argc < 4) {
    fprintf(stderr, "You must specify m, n & k\n");
    return 1;
  }
  m = std::atoi(argv[1]);
  n = std::atoi(argv[2]);
  k = std::atoi(argv[3]);

#ifndef MEASURE_ERROR
  /*        Number of repetitions                                 */

  int nreps = 1;
  if (argc == 5)
    nreps = std::atoi(argv[4]);
  nreps = nreps ? nreps : 1;
#else
  char *outputFile = argv[4];
  FILE *outputFD = fopen(outputFile, "wb");

  if (!outputFD) {
    fprintf(stderr, "Failed to open output file\n");
    return 1;
  }
#endif

  /*       Get input data                                        */

  if (transA == CblasNoTrans) {
    rmaxa = m + 1;
    cmaxa = k;
    ma = m;
    na = k;
  } else {
    rmaxa = k + 1;
    cmaxa = m;
    ma = k;
    na = m;
  }
  if (transB == CblasNoTrans) {
    rmaxb = k + 1;
    cmaxb = n;
    mb = k;
    nb = n;
  } else {
    rmaxb = n + 1;
    cmaxb = k;
    mb = n;
    nb = k;
  }
  rmaxc = m + 1;
  cmaxc = n;
  a = (double *)mkl_calloc(rmaxa * cmaxa, sizeof(double), 64);
  b = (double *)mkl_calloc(rmaxb * cmaxb, sizeof(double), 64);
  c = (double *)mkl_calloc(rmaxc * cmaxc, sizeof(double), 64);
  if (a == NULL || b == NULL || c == NULL) {
    printf("\n Can't allocate memory for arrays\n");
    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
    return 1;
  }

  if (layout == CblasRowMajor) {
    lda = cmaxa;
    ldb = cmaxb;
    ldc = cmaxc;
  } else {
    lda = rmaxa;
    ldb = rmaxb;
    ldc = rmaxc;
  }

  /*              Get random numbers for A, B & C                */
  srand(static_cast<unsigned>(0xDEADBEEF));

#ifndef RANDOM_INPUTS
  /*              Realistic weights extracted from YOLO           */
  int weights_size;
  float * weights = readFromFile("weights.bin", &weights_size);


  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      a[i * lda + j] = static_cast<double>(weights [ ( i * lda + j ) %weights_size ]);

  for (int i = 0; i < k; ++i)
    for (int j = 0; j < n; ++j)
      b[i * ldb + j] = static_cast<double>(weights [ ( i * ldb + j ) %weights_size ]);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      c[i * ldc + j] = static_cast<double>(weights [ ( i * ldc + j ) %weights_size ]);

  free(weights);
#else
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      a[i * lda + j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / 5);

  for (int i = 0; i < k; ++i)
    for (int j = 0; j < n; ++j)
      b[i * ldb + j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / 5);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      c[i * ldc + j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / 5);
#endif

  alpha = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / 5);
  beta = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / 5);

  /*      Call SGEMM subroutine ( C Interface )                  */

#ifdef DEBUG
  printf("================ C INPUT ================\n");
  printMatrix(c, m, n);
#endif

  // number of flops of a single run
  double gflops = 2 * m * n * k / 1e9;
  // number of bytes on a single run
  /*                         A                       B                        C
   *                /                  \   /                   \  /                     \ */
  double gbytes = (sizeof(double) * m * k + sizeof(double) * k * n + sizeof(double) * m * n) /
                  1e9;

#ifndef MEASURE_ERROR
  /*                    WARM-UP EXECUTIONS                    */
  for (int count = 0 ; count < 4 ; ++count)
    cblas_dgemm(layout, transA, transB, m, n, k, alpha,
                  a, lda, b, ldb, beta, c, ldc);

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  for (int count = 0; count < nreps; ++count)
    cblas_dgemm(layout, transA, transB, m, n, k, alpha,
                  a, lda, b, ldb, beta, c, ldc);
  end = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  // time in seconds
  auto time = dur.count() / 1e9;

  printf("dgemm, M = %lld N = %lld K = %lld, GFlops = %lf, GB/s = %lf , execution_time = %lf s\n", m, n, k,
         gflops * nreps / time, gbytes * nreps / time, time);
#else
    cblas_dgemm(layout, transA, transB, m, n, k, alpha,
                  a, lda, b, ldb, beta, c, ldc);
 
    // Store the result of the C matrix in an output file

    // First, write the size of each element in the binary file
    fputc(sizeof(double), outputFD);
    /* 
     * Then, write the type of element:
     * |  0 - FLOAT/DOUBLE (refer to size) 
     * |  1 - FLOAT_16
     * |  2 - INT
    */
    fputc(0, outputFD);

    // write all elements of the output array
    fwrite(c, sizeof(double), m * n, outputFD);

    fclose(outputFD);
#endif

  /*       Print output data                                     */
#ifdef DEBUG
  printf("================ C OUTPUT ================\n");
  printMatrix(c, m, n);
#endif

  mkl_free(a);
  mkl_free(b);
  mkl_free(c);

  return 0;
}

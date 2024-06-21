#include "common.h"
#include <chrono>
#include <cstdio>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MKL_INT m, n, k;
  MKL_INT lda, ldb, ldc;
  MKL_INT rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
  float alpha, beta;
  MKL_F16 *a, *b;
  float *c;
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

  /*        Number of repetitions                                 */

  int nreps = 1;
  if (argc == 5)
    nreps = std::atoi(argv[4]);
  nreps = nreps ? nreps : 1;

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
  a = (MKL_F16 *)mkl_calloc(rmaxa * cmaxa, sizeof(MKL_F16), 64);
  b = (MKL_F16 *)mkl_calloc(rmaxb * cmaxb, sizeof(MKL_F16), 64);
  c = (float *)mkl_calloc(rmaxc * cmaxc, sizeof(float), 64);
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

  /*              Count casting time                            */
  std::chrono::time_point<std::chrono::high_resolution_clock> start_casting;
  start_casting = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      a[i * lda + j] =
          f2h(static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20));

  for (int i = 0; i < k; ++i)
    for (int j = 0; j < n; ++j)
      b[i * ldb + j] =
          f2h(static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20));

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      c[i * ldc + j] =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20);

  alpha = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20);
  beta = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20);

  auto casting_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start_casting);

  /*      Call SGEMM subroutine ( C Interface )                  */

#ifdef DEBUG
  printf("================ C INPUT ================\n");
  printMatrix(c, m, n);
#endif

  // number of flops of a single run
  double gflops = 2 * m * n * k / 1e9;
  // number of bytes on a single run
  /*                          A                         B                         C
   *                /                    \   /                     \   /                    \ */
  double gbytes = (sizeof(MKL_F16) * m * k + sizeof(MKL_F16) * k * n + sizeof(float) * m * n) / 1e9;

  /*                    WARM-UP EXECUTIONS                    */
  for (int count = 0; count < 4; ++count)
    cblas_gemm_f16f16f32(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb,
                         beta, c, ldc);

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  for (int count = 0; count < nreps; ++count)
    cblas_gemm_f16f16f32(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb,
                         beta, c, ldc);
  end = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  // time in seconds
  auto time = dur.count() / 1e9;

#ifdef DEBUG
  MKLVersion Version;
  mkl_get_version(&Version);
  printf("Major version:           %d\n", Version.MajorVersion);
  printf("Minor version:           %d\n", Version.MinorVersion);
  printf("Update version:          %d\n", Version.UpdateVersion);
  printf("Product status:          %s\n", Version.ProductStatus);
  printf("Build:                   %s\n", Version.Build);
  printf("Platform:                %s\n", Version.Platform);
#endif

  printf("f16f16f32 , M = %lld N = %lld K = %lld , GFlops = %lf , GB/s = %lf , "
         "execution_time_with_casting = %lf s\n",
         m, n, k, gflops * nreps / time, gbytes * nreps / time,
         (dur + casting_dur).count() / 1e9);

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

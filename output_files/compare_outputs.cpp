#include "../common.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

std::vector<double> openFile(const char *fileName) {
  FILE *fd = fopen(fileName, "rb");

  std::vector<double> vec;
  if (!fd)
    return vec;

  int size_el = fgetc(fd);
  int type = fgetc(fd);

  if (size_el == EOF || type == EOF) {
    fclose(fd);
    return vec;
  }

  double *arr;

  fseek(fd, 0L, SEEK_END);

  // remove the two bytes we extracted first before calculating
  long num_elements = (ftell(fd) - 2) / size_el;

  arr = (double *)malloc(sizeof(double) * num_elements);

  // return to the start of the array
  fseek(fd, 2L, SEEK_SET);

  // fill array with doubles
  if (size_el == sizeof(double)) {
    if (fread(arr, size_el, num_elements, fd) != num_elements) {
      free(arr);
      fclose(fd);
      return vec;
    }

  }
  // 2 bytes -> float_16
  else if (size_el == 2) {
    for (int i = 0; i < num_elements; ++i) {
      MKL_F16 f16_el;
      if (fread(&f16_el, size_el, 1, fd) != 1) {
        free(arr);
        fclose(fd);
        return vec;
      }

      arr[i] = h2f(f16_el);
    }
  } else
    switch (type) {
    case 0: {
      for (int i = 0; i < num_elements; ++i) {
        float fl_el;
        if (fread(&fl_el, sizeof(float), 1, fd) != 1) {
          free(arr);
          fclose(fd);
          return vec;
        }

        arr[i] = fl_el;
      }
      break;
    }
    case 2: {
      for (int i = 0; i < num_elements; ++i) {
        int int_el;
        if (fread(&int_el, sizeof(int), 1, fd) != 1) {
          free(arr);
          fclose(fd);
          return vec;
        }

        arr[i] = (double)int_el;
      }
      break;
    }
    }
  vec = std::vector<double>(arr, arr + num_elements);
  free(arr);
  fclose(fd);
  return vec;
}

// calculate WAPE from output
double WAPE(std::vector<double> &original, std::vector<double> &relaxed) {
  if (original.size() != relaxed.size())
    return 1.;

  double error = 0.0;
  double den = 0.0;
  for (double val : original) {
    den += val;
  }

  for (int i = 0; i < original.size(); i++) {
    double a = original[i];
    double b = relaxed[i];
    error += std::abs(a - b) / den;
  }

  return error;
}

// calculate how many inputs are wrong,
// without caring for the absolut error value
double WIAP(std::vector<double> &original, std::vector<double> &relaxed) {
  if (original.size() != relaxed.size())
    return 1.;

  unsigned wrong = 0;

  for (int i = 0; i < original.size(); i++) {
    if (original[i] != relaxed[i])
      wrong++;
  }

  return (double)wrong / original.size();
}

double RMSPE(std::vector<double> &original, std::vector<double> &relaxed) {
  if (original.size() != relaxed.size())
    return 1.;

  int n = original.size();

  double sum = .0;
  for (int i = 0; i < n; i++) {
    auto rel_i = (original[i] - relaxed[i]) / original[i]; // rel_i
    sum += rel_i * rel_i;
  }

  return sqrt(sum / n);
}

double RMSE(std::vector<double> &original, std::vector<double> &relaxed) {
  if (original.size() != relaxed.size())
    return 1.;

  int n = original.size();

  double sum = .0;
  for (int i = 0; i < n; i++) {
    auto rel_i = (relaxed[i] - original[i]);
    sum += rel_i * rel_i;
  }

  return sqrt(sum / n);
}

using namespace std;

const vector<string> strvector = {"bf16bf16f32", "f16f16f32", "s8u8s32",
                                  "s16s16s32",   "hgemm",     "sgemm"};

int main(int argc, char *argv[]) {
  int type, size;

  const char *matrix_sizes = argv[1];

  char fileName[40];
  sprintf(fileName, "output_dgemm_%s.bin", matrix_sizes);

  auto original = openFile(fileName);

  if (!original.size()) {
    fprintf(stderr, "dgemm array couldnt be loaded\n");
    return 1;
  }

  for (auto bench : strvector) {
    sprintf(fileName, "output_%s_%s.bin", bench.c_str(), matrix_sizes);

    std::vector<double> approx = openFile(fileName);
    if (!approx.size()) {
      fprintf(stderr, "%s array couldnt be loaded\n", bench.c_str());
      return 1;
    }

    printf("%s,%s,%lf,%lf\n", bench.c_str(), matrix_sizes,
           RMSPE(original, approx), RMSE(original, approx));
  }
}

#include "zyazeva_s_gauss_jordan_elimination/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

ZyazevaSGaussJordanElSEQ::ZyazevaSGaussJordanElSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<float>();
}

bool ZyazevaSGaussJordanElSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  size_t n = matrix.size();

  for (const auto &row : matrix) {
    if (row.size() != n + 1) {
      return false;
    }
  }

  return true;
}

bool ZyazevaSGaussJordanElSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<float>();
  return true;
}

bool ZyazevaSGaussJordanElSEQ::RunImpl() {
  std::vector<std::vector<float>> a = GetInput();
  int n = a.size();

  int flag = 0;

  for (int i = 0; i < n; i++) {
    if (std::abs(a[i][i]) < 1e-7f) {
      int c = 1;
      while ((i + c) < n && std::abs(a[i + c][i]) < 1e-7f) {
        c++;
      }

      if ((i + c) == n) {
        flag = 1;
        break;
      }

      for (int k = 0; k <= n; k++) {
        std::swap(a[i][k], a[i + c][k]);
      }
    }

    float pivot = a[i][i];
    for (int k = i; k <= n; k++) {
      a[i][k] /= pivot;
    }

    for (int j = 0; j < n; j++) {
      if (j != i) {
        float factor = a[j][i];
        for (int k = i; k <= n; k++) {
          a[j][k] -= factor * a[i][k];
        }
      }
    }
  }

  if (flag == 0) {
    std::vector<float> solutions(n);

    for (int i = 0; i < n; i++) {
      solutions[i] = a[i][n];
    }

    GetOutput() = solutions;

    return true;
  } else {
    GetOutput() = std::vector<float>();
    return false;
  }
}

bool ZyazevaSGaussJordanElSEQ::PostProcessingImpl() {
  const auto &solutions = GetOutput();
  return !solutions.empty();
}

}  // namespace zyazeva_s_gauss_jordan_elimination

#include "zyazeva_s_gauss_jordan_elimination/seq/include/ops_seq.hpp"

#include <cfloat>
#include <cmath>

#include "util/include/util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

ZyazevaSGaussJordanElSEQ::ZyazevaSGaussJordanElSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput() = std::move(temp);
  GetOutput() = std::vector<float>();
}

bool ZyazevaSGaussJordanElSEQ::ValidationImpl() {
  const auto& matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  std::size_t n = matrix.size();

  if (!std::all_of(matrix.begin(), matrix.end(), [n](const auto& row) { return row.size() == n + 1; })) {
    return false;
  }
  return true;
}

bool ZyazevaSGaussJordanElSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<float>();
  return true;
}

namespace {

bool kFindAndSwapPivotRow(std::vector<std::vector<float>>& a, int i, int n, float epsilon) {
  if (std::abs(a[i][i]) < epsilon) {
    int c = 1;
    while ((i + c) < n && std::abs(a[i + c][i]) < epsilon) {
      c++;
    }

    if ((i + c) == n) {
      return false;
    }

    for (int k = 0; k <= n; k++) {
      std::swap(a[i][k], a[i + c][k]);
    }
  }
  return true;
}

void kNormalizeCurrentRow(std::vector<std::vector<float>>& a, int i, int n) {
  float pivot = a[i][i];
  for (int k = i; k <= n; k++) {
    a[i][k] /= pivot;
  }
}

void kEliminateColumn(std::vector<std::vector<float>>& a, int i, int n) {
  for (int j = 0; j < n; j++) {
    if (j != i) {
      float factor = a[j][i];
      for (int k = i; k <= n; k++) {
        a[j][k] -= factor * a[i][k];
      }
    }
  }
}

std::vector<float> ExtractSolutions(const std::vector<std::vector<float>>& a, int n) {
  std::vector<float> solutions(n);
  for (int i = 0; i < n; i++) {
    solutions[i] = a[i][n];
  }
  return solutions;
}

}  // namespace

bool ZyazevaSGaussJordanElSEQ::RunImpl() {
  constexpr float kEpsilon = 1e-7F;

  std::vector<std::vector<float>> a = GetInput();
  int n = static_cast<int>(a.size());

  for (int i = 0; i < n; i++) {
    if (!kFindAndSwapPivotRow(a, i, n, kEpsilon)) {
      GetOutput() = std::vector<float>();
      return false;
    }

    kNormalizeCurrentRow(a, i, n);
    kEliminateColumn(a, i, n);
  }

  std::vector<float> solutions = ExtractSolutions(a, n);
  GetOutput() = solutions;

  return true;
}

bool ZyazevaSGaussJordanElSEQ::PostProcessingImpl() {
  const auto& solutions = GetOutput();
  return !solutions.empty();
}

}  // namespace zyazeva_s_gauss_jordan_elimination

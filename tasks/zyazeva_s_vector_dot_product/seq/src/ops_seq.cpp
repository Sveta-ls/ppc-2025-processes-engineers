#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

ZyazevaSVecDotProductSEQ::ZyazevaSVecDotProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ZyazevaSVecDotProductSEQ::ValidationImpl() {
  auto &input = GetInput();
  if (input.size() != 2) {
    return false;
  }
  if (input[0].size() != input[1].size()) {
    return false;
  }
  return !input[0].empty();
}

bool ZyazevaSVecDotProductSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProductSEQ::RunImpl() {
  auto &input = GetInput();
  auto &vec1 = input[0];
  auto &vec2 = input[1];

  int dot_product = 0;

  for (size_t i = 0; i < vec1.size(); i++) {
    dot_product += vec1[i] * vec2[i];
  }

  GetOutput() = dot_product;
  return true;
}

bool ZyazevaSVecDotProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product

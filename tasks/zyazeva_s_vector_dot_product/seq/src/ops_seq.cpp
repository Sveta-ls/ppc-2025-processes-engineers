#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

ZyazevaSVecDotProductSEQ::ZyazevaSVecDotProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput() = std::move(temp);
  GetOutput() = 0;
}

bool ZyazevaSVecDotProductSEQ::ValidationImpl() {
  auto &input = GetInput();
  if ((input[0].empty()) || (input[1].empty())) {
    GetOutput() = 0;
    return true;
  }
  return true;
}

bool ZyazevaSVecDotProductSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProductSEQ::RunImpl() {
  auto &input = GetInput();
  auto &vec1 = input[0];
  auto &vec2 = input[1];

  int64_t dot_product = 0;

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

#pragma once

#include <mpi.h>

#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVecDotProduct : public ppc::task::Task<std::vector<std::vector<int>>, long long> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ZyazevaSVecDotProduct(std::vector<std::vector<int>> input)
      : ppc::task::Task<std::vector<std::vector<int>>, long long>() {
    GetInput() = std::move(input);
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_input1_, local_input2_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  int local_size_;
  long long result{};
  int world_size{};
  int world_rank{};
};

}  // namespace zyazeva_s_vector_dot_product

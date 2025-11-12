#pragma once

#include <mpi.h>

#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVecDotProduct : public BaseTask {
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
};

}  // namespace zyazeva_s_vector_dot_product

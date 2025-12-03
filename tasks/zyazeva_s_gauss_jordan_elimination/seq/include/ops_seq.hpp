#pragma once

#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

class ZyazevaSGaussJordanEiSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyazevaSGaussJordanEiSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyazeva_s_gauss_jordan_elimination

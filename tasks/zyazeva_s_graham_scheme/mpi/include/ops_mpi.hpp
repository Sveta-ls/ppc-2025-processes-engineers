#pragma once

#include "zyazeva_s_graham_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZyazevaSGrahamSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyazeva_s_graham_scheme

#include <gtest/gtest.h>

#include "zyazeva_s_graham_scheme/common/include/common.hpp"
#include "zyazeva_s_graham_scheme/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_graham_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZyazevaSGrahamSchemePerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZyazevaSGrahamSchemeMPI, ZyazevaSGrahamSchemeSEQ>(PPC_SETTINGS_zyazeva_s_graham_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSGrahamSchemePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyazevaSGrahamSchemePerfTest, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_graham_scheme

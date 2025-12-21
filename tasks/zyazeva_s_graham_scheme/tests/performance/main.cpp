#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <random>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_graham_scheme/common/include/common.hpp"
#include "zyazeva_s_graham_scheme/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_graham_scheme/seq/include/ops_seq.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemeLargePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data_;

  void SetUp() override {
    constexpr int kNumPoints = 50000000;
    input_data_.resize(kNumPoints);

    for (int i = 0; i < kNumPoints; ++i) {
      input_data_[i].x = i % 5000;
      input_data_[i].y = (i * 3) % 5000;
    }
  }

  bool CheckTestOutputData(OutType& output_data) final {  // NOLINT
    if (output_data.size() < 3) {
      return true;
    }

    auto Cross = [](const Point& O, const Point& A, const Point& B) {
      return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    };

    const size_t n = output_data.size();
    for (size_t i = 0; i < n; ++i) {
      const Point& a = output_data[i];
      const Point& b = output_data[(i + 1) % n];
      const Point& c = output_data[(i + 2) % n];
      if (Cross(a, b, c) < 0) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZyazevaSGrahamSchemeLargePerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZyazevaSGrahamSchemeMPI, ZyazevaSGrahamSchemeSEQ>(
    PPC_SETTINGS_zyazeva_s_graham_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSGrahamSchemeLargePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyazevaSGrahamSchemeLargePerfTest, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_graham_scheme

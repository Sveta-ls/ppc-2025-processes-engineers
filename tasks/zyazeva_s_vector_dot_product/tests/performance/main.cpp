#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVectorDotProductPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = GenerateLargeVectors(25000000);
  }

  static InType GenerateLargeVectors(size_t size) {
    std::vector<std::vector<int32_t>> vectors(2);
    vectors[0].resize(size);
    vectors[1].resize(size);

    for (size_t i = 0; i < size; ++i) {
      vectors[0][i] = static_cast<int32_t>(1 + (i % 100));
      vectors[1][i] = static_cast<int32_t>(1 + ((i * 7) % 100));
    }

    return vectors;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int64_t expected_res = 0;
    const auto &left_vec = input_data_[0];
    const auto &right_vec = input_data_[1];

    for (size_t i = 0; i < left_vec.size(); i++) {
      expected_res += static_cast<int64_t>(left_vec[i]) * static_cast<int64_t>(right_vec[i]);
    }

    bool result = static_cast<int64_t>(output_data) == expected_res;

    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZyazevaSVectorDotProductPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZyazevaSVecDotProductMPI, ZyazevaSVecDotProductSEQ>(
    PPC_SETTINGS_zyazeva_s_vector_dot_product);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSVectorDotProductPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyazevaSVectorDotProductPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_vector_dot_product

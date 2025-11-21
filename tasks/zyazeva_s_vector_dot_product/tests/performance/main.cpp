#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVectorDotProductPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::string kInputFilename_ =
      "/workspaces/ppc-2025-processes-engineers-1/tasks/zyazeva_s_vector_dot_product/data/input.txt";
  InType input_data_;

  void SetUp() override {
    input_data_ = LoadVectorsFromFile(kInputFilename_);
    input_data_ = LoadVectorsFromFile(kInputFilename_);
  }

  static InType LoadVectorsFromFile(const std::string &filename) {
    std::vector<std::vector<int32_t>> vectors(2);
    std::ifstream file(filename);

    if (!file.is_open()) {
      return vectors;
    }

    std::string line;
    int line_count = 0;

    while (std::getline(file, line)) {
      std::ranges::replace(line, ',', ' ');
      std::istringstream iss(line);
      std::vector<int32_t> vec;
      int32_t value = 0;

      while (iss >> value) {
        vec.push_back(value);
      }

      if (line_count < 2) {
        vectors[line_count] = vec;
      } else {
        break;
      }

      line_count++;
    }

    file.close();
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

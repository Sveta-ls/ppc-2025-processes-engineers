#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"
#include "zyazeva_s_gauss_jordan_elimination/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_gauss_jordan_elimination/seq/include/ops_seq.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

class ZyazevaSGaussJordanFuncTestsSEQ : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{2, 1, 5}, {1, 2, 4}};
        expected_solutions_ = {2.0f, 1.0f};
        break;

      case 1:
        input_data_ = {{1.0f, 1.0f, 1.0f, 6.0f}, {2.0f, 1.0f, 3.0f, 13.0f}, {1.0f, 2.0f, 1.0f, 8.0f}};
        expected_solutions_ = {1.0f, 2.0f, 3.0f};
        break;

      case 2:
        input_data_ = {{3.0f, 0.0f, 0.0f, 6.0f}, {0.0f, 2.0f, 0.0f, 8.0f}, {0.0f, 0.0f, 5.0f, 15.0f}};
        expected_solutions_ = {2.0f, 4.0f, 3.0f};
        break;

      case 3:
        input_data_ = {{2.0f, 1.0f, 3.0f, 12.0f}, {0.0f, 3.0f, 2.0f, 9.0f}, {0.0f, 0.0f, 4.0f, 12.0f}};
        expected_solutions_ = {1.0f, 1.0f, 3.0f};
        break;

      case 4:
        input_data_ = {{1.0f, 0.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f, 7.0f}, {0.0f, 0.0f, 1.0f, 9.0f}};
        expected_solutions_ = {5.0f, 7.0f, 9.0f};
        break;

      case 7:
        input_data_ = {{1.0f, 1.0f, 1.0f, 1.0f, 10.0f},
                       {1.0f, 2.0f, 3.0f, 4.0f, 30.0f},
                       {2.0f, 1.0f, 4.0f, 3.0f, 25.0f},
                       {3.0f, 4.0f, 1.0f, 2.0f, 20.0f}};
        expected_solutions_ = {1.0f, 2.0f, 3.0f, 4.0f};
        break;

      default:
        input_data_ = {{5.0f, 15.0f}};
        expected_solutions_ = {3.0f};
        break;
    }
  }

  auto CheckTestOutputData(OutType &output_data) -> bool final {  // NOLINT
    const float epsilon = 1e-4f;

    if (should_fail_) {
      return output_data.empty() || output_data.size() != expected_solutions_.size();
    }
    if (output_data.size() != expected_solutions_.size()) {
      return false;
    }

    for (size_t i = 0; i < expected_solutions_.size(); ++i) {
      if (std::abs(output_data[i] - expected_solutions_[i]) > epsilon) {
        return false;
      }
    }

    return true;
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_solutions_;
  bool should_fail_ = false;
};

class ZyazevaSGaussJordanFuncTestsMPI : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{2.0f, 1.0f, 5.0f}, {1.0f, 2.0f, 4.0f}};
        expected_solutions_ = {2.0f, 1.0f};
        break;

      case 1:
        input_data_ = {{1.0f, 1.0f, 1.0f, 6.0f}, {2.0f, 1.0f, 3.0f, 13.0f}, {1.0f, 2.0f, 1.0f, 8.0f}};
        expected_solutions_ = {1.0f, 2.0f, 3.0f};
        break;

      case 2:
        input_data_ = {{3.0f, 0.0f, 0.0f, 6.0f}, {0.0f, 2.0f, 0.0f, 8.0f}, {0.0f, 0.0f, 5.0f, 15.0f}};
        expected_solutions_ = {2.0f, 4.0f, 3.0f};
        break;

      case 3:
        input_data_ = {{2.0f, 1.0f, 3.0f, 12.0f}, {0.0f, 3.0f, 2.0f, 9.0f}, {0.0f, 0.0f, 4.0f, 12.0f}};
        expected_solutions_ = {1.0f, 1.0f, 3.0f};
        break;

      case 4:
        input_data_ = {{1.0f, 0.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f, 7.0f}, {0.0f, 0.0f, 1.0f, 9.0f}};
        expected_solutions_ = {5.0f, 7.0f, 9.0f};
        break;

      case 7:
        input_data_ = {{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
                       {0.0f, 1.0f, 0.0f, 0.0f, 2.0f},
                       {0.0f, 0.0f, 1.0f, 0.0f, 3.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f, 4.0f}};
        expected_solutions_ = {1.0f, 2.0f, 3.0f, 4.0f};
        break;

      default:
        input_data_ = {{5.0f, 15.0f}};
        expected_solutions_ = {3.0f};
        break;
    }
  }

  auto CheckTestOutputData(OutType &output_data) -> bool final {  // NOLINT
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
      const float epsilon = 1e-4f;

      if (output_data.size() != expected_solutions_.size()) {
        return false;
      }

      for (size_t i = 0; i < expected_solutions_.size(); ++i) {
        if (std::abs(output_data[i] - expected_solutions_[i]) > epsilon) {
          return false;
        }
      }
    }

    return true;
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_solutions_;
};

namespace {

TEST_P(ZyazevaSGaussJordanFuncTestsSEQ, GaussJordanTestSEQ) {
  ExecuteTest(GetParam());
}

TEST_P(ZyazevaSGaussJordanFuncTestsMPI, GaussJordanTestMPI) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(0, "simple_2x2"),      std::make_tuple(1, "system_3x3"),
    std::make_tuple(2, "diagonal_matrix"), std::make_tuple(3, "triangular_matrix"),
    std::make_tuple(4, "identity_matrix"), std::make_tuple(5, "inconsistent_system"),
    std::make_tuple(6, "singular_matrix")};

const auto kTestTasksListSEQ = ppc::util::AddFuncTask<ZyazevaSGaussJordanElSEQ, InType>(
    kTestParam, PPC_SETTINGS_zyazeva_s_gauss_jordan_elimination);

const std::array<TestType, 7> kTestParamMPI = {
    std::make_tuple(0, "simple_2x2"),       std::make_tuple(1, "system_3x3"),
    std::make_tuple(2, "diagonal_matrix"),  std::make_tuple(3, "triangular_matrix"),
    std::make_tuple(4, "identity1_matrix"), std::make_tuple(7, "system_4x4"),
    std::make_tuple(8, "simple_1x1")};

const auto kTestTasksListMPI = ppc::util::AddFuncTask<ZyazevaSGaussJordanElMPI, InType>(
    kTestParamMPI, PPC_SETTINGS_zyazeva_s_gauss_jordan_elimination);

const auto kGtestValuesSEQ = ppc::util::ExpandToValues(kTestTasksListSEQ);
const auto kGtestValuesMPI = ppc::util::ExpandToValues(kTestTasksListMPI);

const auto kPerfTestNameSEQ = ZyazevaSGaussJordanFuncTestsSEQ::PrintFuncTestName<ZyazevaSGaussJordanFuncTestsSEQ>;
const auto kPerfTestNameMPI = ZyazevaSGaussJordanFuncTestsMPI::PrintFuncTestName<ZyazevaSGaussJordanFuncTestsMPI>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTestsSEQ, ZyazevaSGaussJordanFuncTestsSEQ, kGtestValuesSEQ, kPerfTestNameSEQ);

INSTANTIATE_TEST_SUITE_P(GaussJordanTestsMPI, ZyazevaSGaussJordanFuncTestsMPI, kGtestValuesMPI, kPerfTestNameMPI);

}  // namespace

}  // namespace zyazeva_s_gauss_jordan_elimination

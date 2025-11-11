#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

// Базовый класс для SEQ тестов (без MPI)
class ZyazevaRunFuncTestsSEQ : public ppc::util::BaseRunFuncTests<InType, long long, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{1, 2, 3}, {4, 5, 6}};
        expected_output_ = 32;
        break;
      case 1:
        input_data_ = {{5}, {10}};
        expected_output_ = 50;
        break;
      case 2:
        input_data_ = {{2, 2, 2}, {3, 3, 3}};
        expected_output_ = 18;
        break;
      case 3:
        input_data_ = {{100, 200}, {300, 400}};
        expected_output_ = 110000;
        break;
      default:
        input_data_ = {{1, 2}, {3, 4}};
        expected_output_ = 11;
        break;
    }
  }

  auto CheckTestOutputData(long long &output_data) -> bool final {  // NOLINT
    return (expected_output_ == output_data);
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  long long expected_output_{};
};

// Класс для MPI тестов (с MPI)
class ZyazevaRunFuncTestsMPI : public ppc::util::BaseRunFuncTests<InType, long long, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{1, 2, 3}, {4, 5, 6}};
        expected_output_ = 32;
        break;
      case 1:
        input_data_ = {{5}, {10}};
        expected_output_ = 50;
        break;
      case 2:
        input_data_ = {{2, 2, 2}, {3, 3, 3}};
        expected_output_ = 18;
        break;
      case 3:
        input_data_ = {{100, 200}, {300, 400}};
        expected_output_ = 110000;
        break;
      default:
        input_data_ = {{1, 2}, {3, 4}};
        expected_output_ = 11;
        break;
    }
  }

  auto CheckTestOutputData(long long &output_data) -> bool final {  // NOLINT
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
      return (expected_output_ == output_data);
    } else {
      return true;
    }
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  long long expected_output_{};
};

namespace {

// SEQ тесты
TEST_P(ZyazevaRunFuncTestsSEQ, DotProductTestSEQ) {  // NOLINT
  ExecuteTest(GetParam());
}

// MPI тесты
TEST_P(ZyazevaRunFuncTestsMPI, DotProductTestMPI) {  // NOLINT
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(0, "simple_vectors"),
    std::make_tuple(1, "single_element"),
    std::make_tuple(2, "all_equal"), 
    std::make_tuple(3, "large_values")
};

// SEQ задачи
const auto kTestTasksListSEQ = 
    ppc::util::AddFuncTask<ZyazevaSVecDotProductSEQ, InType>(kTestParam, PPC_SETTINGS_zyazeva_s_vector_dot_product);

// MPI задачи
const auto kTestTasksListMPI = 
    ppc::util::AddFuncTask<ZyazevaSVecDotProduct, InType>(kTestParam, PPC_SETTINGS_zyazeva_s_vector_dot_product);

const auto kGtestValuesSEQ = ppc::util::ExpandToValues(kTestTasksListSEQ);
const auto kGtestValuesMPI = ppc::util::ExpandToValues(kTestTasksListMPI);

const auto kPerfTestNameSEQ = ZyazevaRunFuncTestsSEQ::PrintFuncTestName<ZyazevaRunFuncTestsSEQ>;
const auto kPerfTestNameMPI = ZyazevaRunFuncTestsMPI::PrintFuncTestName<ZyazevaRunFuncTestsMPI>;

// SEQ тест suite
INSTANTIATE_TEST_SUITE_P(  // NOLINT
    VectorDotProductTestsSEQ, ZyazevaRunFuncTestsSEQ, kGtestValuesSEQ, kPerfTestNameSEQ);

// MPI тест suite  
INSTANTIATE_TEST_SUITE_P(  // NOLINT
    VectorDotProductTestsMPI, ZyazevaRunFuncTestsMPI, kGtestValuesMPI, kPerfTestNameMPI);

}  // namespace

}  // namespace zyazeva_s_vector_dot_product

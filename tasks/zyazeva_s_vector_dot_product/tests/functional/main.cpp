#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "util/include/func_test_util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVecDotFuncTests : public testing::TestWithParam<TestType> {
 protected:
  void SetUp() override {
    auto [test_case, name] = GetParam();

    if (test_case == 1) {
      // Большие векторы с последовательными числами
      std::vector<int> vec1(1000), vec2(1000);
      std::iota(vec1.begin(), vec1.end(), 1);     // 1, 2, 3, ..., 1000
      std::iota(vec2.begin(), vec2.end(), 1001);  // 1001, 1002, ..., 2000

      int n = 1000;
      expected_result_ = (n * (n + 1) * (2 * n + 1)) / 6 + 1000 * (n * (n + 1)) / 2;
      input_data_ = {vec1, vec2};

    } else if (test_case == 2) {
      // Векторы с повторяющимся паттерном
      std::vector<int> vec1(500), vec2(500);
      for (size_t i = 0; i < 500; i++) {
        vec1[i] = (i % 2 == 0) ? 2 : -1;
        vec2[i] = (i % 3 == 0) ? 3 : -2;
      }

      expected_result_ = 0;
      for (size_t i = 0; i < 500; i++) {
        int val1 = (i % 2 == 0) ? 2 : -1;
        int val2 = (i % 3 == 0) ? 3 : -2;
        expected_result_ += val1 * val2;
      }
      input_data_ = {vec1, vec2};

    } else if (test_case == 3) {
      // Случайные большие векторы (детерминированные)
      std::vector<int> vec1(800), vec2(800);
      for (size_t i = 0; i < 800; i++) {
        vec1[i] = static_cast<int>(i * 3 + 1);
        vec2[i] = static_cast<int>(i * 2 - 5);
      }

      expected_result_ = 0;
      for (size_t i = 0; i < 800; i++) {
        expected_result_ += vec1[i] * vec2[i];
      }
      input_data_ = {vec1, vec2};
    }
  }

  InType input_data_;
  int expected_result_;
};

// SEQ тесты
TEST_P(ZyazevaSVecDotFuncTests, SequentialTest) {
  auto task = std::make_shared<ZyazevaSVecDotProductSEQ>(input_data_);

  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());

  EXPECT_EQ(task->GetOutput(), expected_result_);
}

// MPI тесты
TEST_P(ZyazevaSVecDotFuncTests, MPITest) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto task = std::make_shared<ZyazevaSVecDotProduct>(input_data_);

  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());

  // Проверяем результат только на процессе 0
  // Другие процессы возвращают 0, но это нормально для MPI
  if (world_rank == 0) {
    EXPECT_EQ(task->GetOutput(), expected_result_);
  } else {
    // На других процессах результат может быть 0 - это ожидаемо
    // Не проверяем результат на не-0 процессах
    SUCCEED();
  }
}

INSTANTIATE_TEST_SUITE_P(VectorTests, ZyazevaSVecDotFuncTests,
                         testing::Values(std::make_tuple(1, "large_sequential"), std::make_tuple(2, "large_pattern"),
                                         std::make_tuple(3, "large_arithmetic")),
                         [](const testing::TestParamInfo<TestType> &info) { return std::get<1>(info.param); });

}  // namespace zyazeva_s_vector_dot_product

#include <gtest/gtest.h>
#include <mpi.h>

#include <numeric>
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
    std::vector<int> vec1, vec2;

    if (test_case == 1) {
      vec1.resize(100000);
      vec2.resize(100000);
      std::iota(vec1.begin(), vec1.end(), 1);
      std::iota(vec2.begin(), vec2.end(), 10001);
    } else if (test_case == 2) {
      vec1.resize(50000);
      vec2.resize(50000);
      for (size_t i = 0; i < 50000; i++) {
        vec1[i] = (i % 2 == 0) ? 2 : -1;
        vec2[i] = (i % 3 == 0) ? 3 : -2;
      }
    } else if (test_case == 3) {
      vec1.resize(80000);
      vec2.resize(80000);
      for (size_t i = 0; i < 80000; i++) {
        vec1[i] = i + 1;
        vec2[i] = i * 2 - 5;
      }
    }

    expected_result_ = 0;
    for (size_t i = 0; i < vec1.size(); i++) {
      expected_result_ += vec1[i] * vec2[i];
    }
    input_data_ = {vec1, vec2};
  }

  InType input_data_;
  int expected_result_;
};

// SEQ тесты - тоже переписываем на стек для консистентности
TEST_P(ZyazevaSVecDotFuncTests, SequentialTest) {
  ZyazevaSVecDotProductSEQ task(input_data_);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), expected_result_);
}

// MPI тесты - используем объект на стеке
TEST_P(ZyazevaSVecDotFuncTests, MPITest) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Создаем объект на стеке
  ZyazevaSVecDotProduct task(input_data_);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  if (world_rank == 0) {
    EXPECT_EQ(task.GetOutput(), expected_result_);
  }

  // Синхронизация перед разрушением
  MPI_Barrier(MPI_COMM_WORLD);

  // Объект будет автоматически разрушен при выходе из scope
}

INSTANTIATE_TEST_SUITE_P(VectorTests, ZyazevaSVecDotFuncTests,
                         testing::Values(std::make_tuple(1, "sequential"), std::make_tuple(2, "pattern"),
                                         std::make_tuple(3, "arithmetic")),
                         [](const auto& info) { return std::get<1>(info.param); });

}  // namespace zyazeva_s_vector_dot_product

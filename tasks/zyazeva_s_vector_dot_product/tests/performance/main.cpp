#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

TEST(PerformanceTest, SequentialSmallVectors) {
  // Используем МАЛЕНЬКИЕ векторы чтобы избежать переполнения
  std::vector<int> vec1(5000), vec2(5000);
  std::iota(vec1.begin(), vec1.end(), 1);     // 1, 2, 3, ..., 1000
  std::iota(vec2.begin(), vec2.end(), 5001);  // 1001, 1002, ..., 2000

  InType input_data = {vec1, vec2};

  auto task = std::make_shared<ZyazevaSVecDotProductSEQ>(input_data);

  // Замеряем время выполнения
  auto start = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "SEQ Execution time: " << duration.count() << " microseconds" << std::endl;
  std::cout << "Result: " << task->GetOutput() << std::endl;

  // Проверяем что результат положительный
  EXPECT_GT(task->GetOutput(), 0);
}

// Простой перфоманс тест для MPI версии с МАЛЕНЬКИМИ векторами
TEST(PerformanceTest, MPISmallVectors) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Используем МАЛЕНЬКИЕ векторы чтобы избежать переполнения
  std::vector<int> vec1(5000), vec2(5000);
  if (world_rank == 0) {
    std::iota(vec1.begin(), vec1.end(), 1);     // 1, 2, 3, ..., 1000
    std::iota(vec2.begin(), vec2.end(), 5001);  // 1001, 1002, ..., 2000
  }

  InType input_data = {vec1, vec2};

  auto task = std::make_shared<ZyazevaSVecDotProduct>(input_data);

  // Замеряем время выполнения
  auto start = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  if (world_rank == 0) {
    std::cout << "MPI Execution time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Result: " << task->GetOutput() << std::endl;

    // Проверяем корректность результата только на процессе 0
    EXPECT_GT(task->GetOutput(), 0);
  }
}

// Тест с РЕАЛЬНО маленькими размерами векторов
TEST(PerformanceTest, DifferentSmallSizes) {
  std::vector<int> sizes = {100, 500, 1000};  // МАЛЕНЬКИЕ размеры

  for (int size : sizes) {
    std::vector<int> vec1(size), vec2(size);

    // Заполняем маленькими числами чтобы избежать переполнения
    for (int i = 0; i < size; i++) {
      vec1[i] = i + 1;         // 1, 2, 3, ..., size
      vec2[i] = size + i + 1;  // size+1, size+2, ..., 2*size
    }

    InType input_data = {vec1, vec2};
    auto task = std::make_shared<ZyazevaSVecDotProductSEQ>(input_data);

    auto start = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(task->Validation());
    EXPECT_TRUE(task->PreProcessing());
    EXPECT_TRUE(task->Run());
    EXPECT_TRUE(task->PostProcessing());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Size " << size << ": " << duration.count() << " microseconds" << std::endl;
    std::cout << "Result for size " << size << ": " << task->GetOutput() << std::endl;

    // Проверяем что результат положительный
    EXPECT_GT(task->GetOutput(), 0);
  }
}

}  // namespace zyazeva_s_vector_dot_product

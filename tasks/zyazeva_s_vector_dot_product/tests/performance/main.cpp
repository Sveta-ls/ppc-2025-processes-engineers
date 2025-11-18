#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

TEST(SimplePerfTest, CompareBothVersions) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int size = 1000000;

  if (rank == 0) {
    std::vector<std::vector<int>> seq_data(2);
    seq_data[0].resize(size);
    seq_data[1].resize(size);

    for (int i = 0; i < size; i++) {
      seq_data[0][i] = i % 50;
      seq_data[1][i] = (i * 2) % 50;
    }

    auto seq_task = std::make_shared<ZyazevaSVecDotProductSEQ>(seq_data);

    auto seq_start = std::chrono::high_resolution_clock::now();
    seq_task->Validation();
    seq_task->PreProcessing();
    seq_task->Run();
    seq_task->PostProcessing();
    auto seq_end = std::chrono::high_resolution_clock::now();

    auto seq_time = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - seq_start);
    std::cout << "SEQ: " << seq_time.count() << '\n';

    auto seq_duration = static_cast<double>(seq_time.count());

    std::vector<std::vector<int>> mpi_data(2);
    mpi_data[0].resize(size);
    mpi_data[1].resize(size);
    for (int i = 0; i < size; i++) {
      mpi_data[0][i] = i % 50;
      mpi_data[1][i] = (i * 2) % 50;
    }

    auto mpi_task = std::make_shared<ZyazevaSVecDotProduct>(mpi_data);

    double mpi_start = MPI_Wtime();
    mpi_task->Validation();
    mpi_task->PreProcessing();
    mpi_task->Run();
    mpi_task->PostProcessing();
    double mpi_end = MPI_Wtime();

    double mpi_duration = (mpi_end - mpi_start) * 1000000.0;
    std::cout << "MPI: " << mpi_duration << '\n';

    if (mpi_duration > 0) {
      std::cout << "Отношение SEQ/MPI: " << (seq_duration / mpi_duration) << '\n';
    }
  } else {
    std::vector<std::vector<int>> mpi_data(2);
    auto mpi_task = std::make_shared<ZyazevaSVecDotProduct>(mpi_data);

    mpi_task->Validation();
    mpi_task->PreProcessing();
    mpi_task->Run();
    mpi_task->PostProcessing();
  }
}

}  // namespace zyazeva_s_vector_dot_product

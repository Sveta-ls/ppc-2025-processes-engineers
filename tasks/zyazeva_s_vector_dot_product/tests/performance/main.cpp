#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

namespace {

std::vector<std::vector<int>> LoadVectorsFromFile(const std::string &filename) {
  std::vector<std::vector<int>> vectors(2);
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::string line;
  int line_count = 0;

  while (std::getline(file, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');

    std::istringstream iss(line);
    std::vector<int> vec;
    int value = 0;

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

}  // namespace

TEST(SimplePerfTest, CompareBothVersions) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::vector<std::vector<int>> data;

    data = LoadVectorsFromFile("tasks/zyazeva_s_vector_dot_product/data/input.txt");

    auto seq_task = std::make_shared<ZyazevaSVecDotProductSEQ>(data);

    auto seq_start = std::chrono::high_resolution_clock::now();
    seq_task->Validation();
    seq_task->PreProcessing();
    seq_task->Run();
    seq_task->PostProcessing();
    auto seq_end = std::chrono::high_resolution_clock::now();

    auto seq_time = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - seq_start);
    std::cout << "SEQ: " << seq_time.count() << " microseconds\n";

    auto seq_duration = static_cast<double>(seq_time.count());

    auto mpi_task = std::make_shared<ZyazevaSVecDotProduct>(data);

    double mpi_start = MPI_Wtime();
    mpi_task->Validation();
    mpi_task->PreProcessing();
    mpi_task->Run();
    mpi_task->PostProcessing();
    double mpi_end = MPI_Wtime();

    double mpi_duration = (mpi_end - mpi_start) * 1000000.0;
    std::cout << "MPI: " << mpi_duration << " microseconds\n";

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

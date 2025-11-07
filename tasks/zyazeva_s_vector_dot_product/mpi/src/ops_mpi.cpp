#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProduct::ValidationImpl() {
  auto& input = GetInput();
  return input.size() == 2 && input[0].size() == input[1].size();
}

bool ZyazevaSVecDotProduct::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto& input = GetInput();
  int total_elements = static_cast<int>(input[0].size());

  // Вычисляем распределение данных
  counts_.resize(world_size);
  displs_.resize(world_size);

  int base_size = total_elements / world_size;
  int remainder = total_elements % world_size;

  for (int i = 0; i < world_size; ++i) {
    counts_[i] = base_size + (i < remainder ? 1 : 0);
    displs_[i] = (i == 0) ? 0 : displs_[i - 1] + counts_[i - 1];
  }

  local_size_ = counts_[world_rank];
  local_input1_.resize(local_size_);
  local_input2_.resize(local_size_);

  return true;
}

bool ZyazevaSVecDotProduct::RunImpl() {
  auto& input = GetInput();

  // Используем векторные операции вместо последовательных Send/Recv
  MPI_Scatterv(input[0].data(), counts_.data(), displs_.data(), MPI_INT, local_input1_.data(), local_size_, MPI_INT, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(input[1].data(), counts_.data(), displs_.data(), MPI_INT, local_input2_.data(), local_size_, MPI_INT, 0,
               MPI_COMM_WORLD);

  int local_result = 0;
  for (int i = 0; i < local_size_; i++) {
    local_result += local_input1_[i] * local_input2_[i];
  }

  // Сбор результатов
  MPI_Reduce(&local_result, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool ZyazevaSVecDotProduct::PostProcessingImpl() {
  if (world_rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput() = 0;
  }
  return true;
}

}  // namespace zyazeva_s_vector_dot_product

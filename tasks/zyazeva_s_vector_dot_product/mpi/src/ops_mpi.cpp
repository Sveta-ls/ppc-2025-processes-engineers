#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProduct::ValidationImpl() {
  SetTypeOfTask(GetStaticTypeOfTask());
  const auto &input = GetInput();
  return input.size() == 2 && input[0].size() == input[1].size();
}

bool ZyazevaSVecDotProduct::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProduct::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const auto &vector1 = input[0];
  const auto &vector2 = input[1];

  int local_size =
      (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) ? static_cast<int>(vector1.size()) : 0;

  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (max_size == 0) {
    GetOutput() = 0;
    return true;
  }

  const auto n = static_cast<size_t>(max_size);
  const size_t base_chunk = n / static_cast<size_t>(size);
  const size_t remainder = n % static_cast<size_t>(size);

  const int remainder_i = static_cast<int>(remainder);
  const bool has_extra = (rank < remainder_i);

  const size_t start = (static_cast<size_t>(rank) * base_chunk) + static_cast<size_t>(std::min(rank, remainder_i));
  const size_t end = start + base_chunk + (has_extra ? 1U : 0U);

  int64_t local_dot = 0;

  if (local_size > 0) {
    const size_t local_n = vector1.size();
    const size_t real_start = std::min(start, local_n);
    const size_t real_end = std::min(end, local_n);

    for (size_t i = real_start; i < real_end; ++i) {
      local_dot += static_cast<int64_t>(vector1[i]) * static_cast<int64_t>(vector2[i]);
    }
  }

  int64_t global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot);
  return true;
}

bool ZyazevaSVecDotProduct::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product

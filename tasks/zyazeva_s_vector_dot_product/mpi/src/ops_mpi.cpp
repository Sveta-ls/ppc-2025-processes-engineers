#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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

  bool has_data = !vector1.empty() && !vector2.empty() && (vector1.size() == vector2.size());
  int local_size = has_data ? static_cast<int>(vector1.size()) : 0;

  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (max_size == 0) {
    GetOutput() = 0;
    return true;
  }

  const auto total_elements = static_cast<size_t>(max_size);
  const auto chunk_size = total_elements / static_cast<size_t>(size);
  const auto remaining_elements = total_elements % static_cast<size_t>(size);

  const size_t start_index =
      (static_cast<size_t>(rank) * chunk_size) + std::min(static_cast<size_t>(rank), remaining_elements);

  const bool needs_more_element = (static_cast<int>(remaining_elements) > static_cast<int>(rank));
  const size_t end_index = start_index + chunk_size + (needs_more_element ? 1UL : 0UL);

  int64_t local_dot_product = 0;

  if (has_data) {
    const size_t actual_start = std::min(start_index, vector1.size());
    const size_t actual_end = std::min(end_index, vector1.size());

    for (size_t i = actual_start; i < actual_end; ++i) {
      local_dot_product += static_cast<int64_t>(vector1[i]) * static_cast<int64_t>(vector2[i]);
    }
  }

  int64_t global_dot_product = 0;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot_product);
  return true;
}

bool ZyazevaSVecDotProduct::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product

#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProductMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.size() == 2 && input[0].size() == input[1].size();
}

bool ZyazevaSVecDotProductMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ZyazevaSVecDotProductMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const auto &vector1 = input[0];
  const auto &vector2 = input[1];

  if (vector1.empty() || vector2.empty()) {
    GetOutput() = 0;
    return true;
  }

  const size_t total_elements = vector1.size();

  size_t chunk_size = total_elements / size;
  size_t start = rank * chunk_size;
  size_t end = (rank == size - 1) ? total_elements : start + chunk_size;

  int64_t local_dot_product = 0;
  for (size_t i = start; i < end; ++i) {
    local_dot_product += static_cast<int64_t>(vector1[i]) * static_cast<int64_t>(vector2[i]);
  }

  int64_t global_dot_product = 0;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot_product);

  return true;
}

bool ZyazevaSVecDotProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_vector_dot_product

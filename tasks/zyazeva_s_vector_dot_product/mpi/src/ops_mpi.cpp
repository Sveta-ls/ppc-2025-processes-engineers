#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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

  int total_elements = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    total_elements = static_cast<int>(input[0].size());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elements == 0) {
    GetOutput() = 0;
    return true;
  }

  int chunk_size = total_elements / size;
  int start = rank * chunk_size;
  int end = (rank == size - 1) ? total_elements : start + chunk_size;
  int local_size = end - start;

  std::vector<int32_t> local_vector1(local_size);
  std::vector<int32_t> local_vector2(local_size);

  if (rank == 0) {
    const auto &input = GetInput();
    const auto &vector1_full = input[0];
    const auto &vector2_full = input[1];

    std::copy(vector1_full.begin() + start, vector1_full.begin() + end, local_vector1.begin());
    std::copy(vector2_full.begin() + start, vector2_full.begin() + end, local_vector2.begin());

    for (int i = 1; i < size; i++) {
      int i_start = i * chunk_size;
      int i_end = (i == size - 1) ? total_elements : i_start + chunk_size;
      int i_size = i_end - i_start;

      MPI_Send(vector1_full.data() + i_start, i_size, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(vector2_full.data() + i_start, i_size, MPI_INT, i, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_vector1.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_vector2.data(), local_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int64_t local_dot_product = 0;
  for (int i = 0; i < local_size; ++i) {
    local_dot_product += static_cast<int64_t>(local_vector1[i]) * static_cast<int64_t>(local_vector2[i]);
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

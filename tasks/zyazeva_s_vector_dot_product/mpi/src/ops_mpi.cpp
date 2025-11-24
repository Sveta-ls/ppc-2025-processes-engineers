#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "zyazeva_s_vector_dot_product/common/include/common.hpp"

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProductMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  bool is_valid = false;
  
  if (rank == 0) {
    const auto &input = GetInput();
    
    if (input.size() >= 2) {
      const auto& vector1 = input[0];
      const auto& vector2 = input[1];
      is_valid = (vector1.size() == vector2.size()) && !vector1.empty() && !vector2.empty();
    } else {
      is_valid = true;
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  
  return is_valid;
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

  int64_t total_elements = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    
    if (input.size() < 2) {
      GetOutput() = 0;
      total_elements = -1; 
    } else {
      const auto& vector1 = input[0];
      const auto& vector2 = input[1];
      
      if (vector1.size() != vector2.size() || vector1.empty() || vector2.empty()) {
        GetOutput() = 0;
        total_elements = -1;
      } else {
        total_elements = static_cast<int64_t>(vector1.size());
      }
    }
  }

<<<<<<< HEAD
  MPI_Bcast(&total_elements, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  if (total_elements == -1) {
    if (rank != 0) GetOutput() = 0;
    return true;
  }

  if (total_elements == 0) {
    if (rank != 0) GetOutput() = 0;
    return true;
  }

  int64_t base_chunk_size = total_elements / size;
  int64_t remainder = total_elements % size;
  
  int64_t local_size = base_chunk_size + (rank < remainder ? 1 : 0);
  int64_t start = 0;
  
  if (rank < remainder) {
    start = rank * (base_chunk_size + 1);
  } else {
    start = remainder * (base_chunk_size + 1) + (rank - remainder) * base_chunk_size;
  }
=======
  int chunk_size = total_elements / size;
  int start = rank * chunk_size;
  int end = (rank == size - 1) ? total_elements : start + chunk_size;
  int local_size = end - start;
>>>>>>> parent of d2a902d (fixed clang-tigy2)

  std::vector<int32_t> local_vector1(local_size);
  std::vector<int32_t> local_vector2(local_size);

  if (rank == 0) {
    const auto &input = GetInput();
    const auto &vector1_full = input[0];
    const auto &vector2_full = input[1];

    std::copy(vector1_full.begin() + start, vector1_full.begin() + start + local_size, local_vector1.begin());
    std::copy(vector2_full.begin() + start, vector2_full.begin() + start + local_size, local_vector2.begin());

    for (int i = 1; i < size; i++) {
<<<<<<< HEAD
      int64_t i_local_size = base_chunk_size + (i < remainder ? 1 : 0);
      int64_t i_start = 0;
      
      if (i < remainder) {
        i_start = i * (base_chunk_size + 1);
      } else {
        i_start = remainder * (base_chunk_size + 1) + (i - remainder) * base_chunk_size;
      }
      MPI_Send(vector1_full.data() + i_start, static_cast<int>(i_local_size), MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(vector2_full.data() + i_start, static_cast<int>(i_local_size), MPI_INT, i, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_vector1.data(), static_cast<int>(local_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_vector2.data(), static_cast<int>(local_size), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
=======
      int i_start = i * chunk_size;
      int i_end = (i == size - 1) ? total_elements : i_start + chunk_size;
      int i_size = i_end - i_start;

      MPI_Send(vector1_full.data() + i_start, i_size, MPI_INT32_T, i, 0, MPI_COMM_WORLD);
      MPI_Send(vector2_full.data() + i_start, i_size, MPI_INT32_T, i, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_vector1.data(), local_size, MPI_INT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_vector2.data(), local_size, MPI_INT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
>>>>>>> parent of d2a902d (fixed clang-tigy2)
  }

  int64_t local_dot_product = 0;
  for (int64_t i = 0; i < local_size; ++i) {
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

#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProduct::ValidationImpl() {
  auto& input = GetInput();
  if (input.size() != 2) {
    return false;
  }
  if (input[0].size() != input[1].size()) {
    return false;
  }
  return true;
}

bool ZyazevaSVecDotProduct::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto& input = GetInput();
  
  if (world_rank == 0) {
    int total_elements = static_cast<int>(input[0].size());
    counts_.resize(world_size);
    
    int delta = total_elements / world_size;
    int remainder = total_elements % world_size;
    
    for (int i = 0; i < world_size; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);
    }
  }
  
  int my_count;
  if (world_rank == 0) {
    my_count = counts_[0];
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&counts_[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&my_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  local_size_ = my_count;
  
  result = 0;
  return true;
}

bool ZyazevaSVecDotProduct::RunImpl() {
  auto& input = GetInput();
  
  local_input1_.resize(local_size_);
  local_input2_.resize(local_size_);
  
  if (world_rank == 0) {
    std::copy(input[0].begin(), input[0].begin() + local_size_, local_input1_.begin());
    std::copy(input[1].begin(), input[1].begin() + local_size_, local_input2_.begin());
    
    int offset = local_size_;
    
    for (int proc = 1; proc < world_size; proc++) {
      int proc_count = counts_[proc];
      MPI_Send(input[0].data() + offset, proc_count, MPI_INT, proc, 0, MPI_COMM_WORLD);
      MPI_Send(input[1].data() + offset, proc_count, MPI_INT, proc, 1, MPI_COMM_WORLD);

      offset += proc_count;
    }
  } else {
    MPI_Recv(local_input1_.data(), local_size_, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_input2_.data(), local_size_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_result = 0;
  for (int i = 0; i < local_size_; i++) {
    local_result += local_input1_[i] * local_input2_[i];
  }

  if (world_rank == 0) {
    result = local_result;
   
    for (int proc = 1; proc < world_size; proc++) {
      int proc_result;
      MPI_Recv(&proc_result, 1, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result += proc_result;
    }
  } else {
    MPI_Send(&local_result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
  }
  
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
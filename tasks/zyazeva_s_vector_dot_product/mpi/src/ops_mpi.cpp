#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace zyazeva_s_vector_dot_product {

bool ZyazevaSVecDotProduct::ValidationImpl() {
  auto& input = GetInput();
  if (input.size() != 2) {
    std::cout << "Validation failed: input size != 2" << std::endl;
    return false;
  }
  if (input[0].size() != input[1].size()) {
    std::cout << "Validation failed: vector sizes don't match" << std::endl;
    return false;
  }
  std::cout << "Validation passed: vectors size = " << input[0].size() << std::endl;
  return true;
}

bool ZyazevaSVecDotProduct::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto& input = GetInput();
  
  std::cout << "Process " << world_rank << ": PreProcessing started, vectors size = " 
            << input[0].size() << std::endl;

  // Процесс 0 вычисляет размеры для всех
  if (world_rank == 0) {
    int total_elements = static_cast<int>(input[0].size());
    counts_.resize(world_size);
    
    int delta = total_elements / world_size;
    int remainder = total_elements % world_size;
    
    std::cout << "Process 0: total_elements = " << total_elements 
              << ", delta = " << delta << ", remainder = " << remainder << std::endl;
    
    for (int i = 0; i < world_size; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);
      std::cout << "Process 0: counts_[" << i << "] = " << counts_[i] << std::endl;
    }
  }
  
  // Каждый процесс получает свой размер
  int my_count;
  if (world_rank == 0) {
    my_count = counts_[0];
    // Отправляем размеры другим процессам
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&counts_[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      std::cout << "Process 0: sent count " << counts_[i] << " to process " << i << std::endl;
    }
  } else {
    MPI_Recv(&my_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "Process " << world_rank << ": received count = " << my_count << std::endl;
  }
  
  // Сохраняем размер для текущего процесса
  local_size_ = my_count;
  
  result = 0;
  return true;
}

bool ZyazevaSVecDotProduct::RunImpl() {
  auto& input = GetInput();
  
  std::cout << "Process " << world_rank << ": RunImpl started, local_size = " << local_size_ << std::endl;

  // Подготовка локальных векторов
  local_input1_.resize(local_size_);
  local_input2_.resize(local_size_);
  
  if (world_rank == 0) {
    // Процесс 0 копирует свою часть напрямую
    std::copy(input[0].begin(), input[0].begin() + local_size_, local_input1_.begin());
    std::copy(input[1].begin(), input[1].begin() + local_size_, local_input2_.begin());
    
    std::cout << "Process 0: copied data, first elements: " 
              << local_input1_[0] << ", " << local_input2_[0] << std::endl;
    
    int offset = local_size_;
    
    // Отправляем данные другим процессам
    for (int proc = 1; proc < world_size; proc++) {
      int proc_count = counts_[proc];
      MPI_Send(input[0].data() + offset, proc_count, MPI_INT, proc, 0, MPI_COMM_WORLD);
      MPI_Send(input[1].data() + offset, proc_count, MPI_INT, proc, 1, MPI_COMM_WORLD);
      std::cout << "Process 0: sent " << proc_count << " elements to process " << proc 
                << " starting from offset " << offset << std::endl;
      offset += proc_count;
    }
  } else {
    // Другие процессы получают свои данные
    MPI_Recv(local_input1_.data(), local_size_, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_input2_.data(), local_size_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "Process " << world_rank << ": received data, first elements: " 
              << local_input1_[0] << ", " << local_input2_[0] << std::endl;
  }

  // ВЫЧИСЛЯЕМ локальное скалярное произведение
  int local_result = 0;
  for (int i = 0; i < local_size_; i++) {
    local_result += local_input1_[i] * local_input2_[i];
  }

  std::cout << "Process " << world_rank << ": local_result = " << local_result 
            << ", calculated from " << local_size_ << " elements" << std::endl;

  // Собираем результаты на процессе 0
  if (world_rank == 0) {
    result = local_result;
    std::cout << "Process 0: initial result = " << result << std::endl;
    
    // Получаем результаты от других процессов
    for (int proc = 1; proc < world_size; proc++) {
      int proc_result;
      MPI_Recv(&proc_result, 1, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result += proc_result;
      std::cout << "Process 0: received result " << proc_result << " from process " << proc 
                << ", total now = " << result << std::endl;
    }
  } else {
    // Отправляем свой результат процессу 0
    MPI_Send(&local_result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    std::cout << "Process " << world_rank << ": sent result " << local_result << " to process 0" << std::endl;
  }
  
  return true;
}

bool ZyazevaSVecDotProduct::PostProcessingImpl() {
  // Записываем результат в выходные данные
  if (world_rank == 0) {
    GetOutput() = result;
    std::cout << "Process 0: FINAL RESULT = " << result << std::endl;
  } else {
    GetOutput() = 0;  // Другие процессы возвращают 0
  }
  return true;
}

}  // namespace zyazeva_s_vector_dot_product
#include "zyazeva_s_gauss_jordan_elimination/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <utility>

#include "util/include/util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

ZyazevaSGaussJordanElMPI::ZyazevaSGaussJordanElMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput() = std::move(temp);
}

bool ZyazevaSGaussJordanElMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;

  if (rank == 0) {
    const auto& matrix = GetInput();

    if (matrix.empty()) {
      is_valid = false;
    } else {
      int n = static_cast<int>(matrix.size());
      is_valid = true;

      for (const auto& row : matrix) {
        if (row.size() != static_cast<std::size_t>(n + 1)) {
          is_valid = false;
          break;
        }
      }
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool ZyazevaSGaussJordanElMPI::PreProcessingImpl() {
  GetOutput() = std::vector<float>();
  return true;
}
namespace {

std::vector<int> DistributeRows(int size, int n) {
  std::vector<int> rows(size);
  int base = n / size;
  int extra = n % size;
  for (int proc = 0; proc < size; ++proc) {
    rows[proc] = base + (proc < extra ? 1 : 0);
  }
  return rows;
}

void CalculateScatterParameters(int size, const std::vector<int>& rows, int width, std::vector<int>& send_counts,
                                std::vector<int>& displs) {
  int offset_rows = 0;
  for (int proc = 0; proc < size; ++proc) {
    send_counts[proc] = rows[proc] * width;
    displs[proc] = offset_rows * width;
    offset_rows += rows[proc];
  }
}

void ScatterMatrix(int rank, const std::vector<double>& flat, const std::vector<int>& send_counts,
                   const std::vector<int>& displs, std::vector<double>& local_matrix, int local_rows, int width) {
  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, send_counts.data(), displs.data(), MPI_DOUBLE, local_matrix.data(),
               local_rows * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int CalculateStartRow(int rank, const std::vector<int>& rows) {
  int start_row = 0;
  for (int proc = 0; proc < rank; ++proc) {
    start_row += rows[proc];
  }
  return start_row;
}

int FindRowOwner(int row_index, int size, const std::vector<int>& rows) {
  int owner = 0;
  int acc = 0;
  for (int proc = 0; proc < size; ++proc) {
    if (row_index >= acc && row_index < acc + rows[proc]) {
      owner = proc;
      break;
    }
    acc += rows[proc];
  }
  return owner;
}

int FindLocalPivotRow(int k, int start_row, int local_rows, const std::vector<double>& local_matrix, int width,
                      double k_eps) {
  int local_found = INT_MAX;
  for (int i = 0; i < local_rows; ++i) {
    int gi = start_row + i;
    if (gi < k) {
      continue;
    }
    if (std::fabs(local_matrix[static_cast<std::size_t>((i * width) + k)]) > k_eps) {
      local_found = gi;
      break;
    }
  }
  return local_found;
}

int FindGlobalPivotRow(int local_found) {
  int global_found = 0;
  MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global_found;
}

void HandleZeroPivot(int rank, int owner, std::vector<double>& pivot, int width) {
  if (rank == owner) {
    std::fill(pivot.begin(), pivot.end(), 0.0);
  }
  MPI_Bcast(pivot.data(), width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
}

void SwapRowsSameOwner(int rank, int owner, int k, int global_found, int start_row, std::vector<double>& local_matrix,
                       int width) {
  if (rank == owner) {
    int lk = k - start_row;
    int lf = global_found - start_row;
    for (int j = 0; j < width; ++j) {
      std::swap(local_matrix[static_cast<std::size_t>(lk * width + j)],
                local_matrix[static_cast<std::size_t>(lf * width + j)]);
    }
  }
}

void SwapRowsDifferentOwners(int rank, int owner, int owner2, int k, int global_found, int start_row,
                             std::vector<double>& local_matrix, int width) {
  std::vector<double> row_k;
  std::vector<double> row_f;

  using diff_t = std::vector<double>::difference_type;

  if (rank == owner) {
    diff_t offset = static_cast<diff_t>((k - start_row)) * static_cast<diff_t>(width);
    row_k.assign(local_matrix.begin() + offset, local_matrix.begin() + offset + static_cast<diff_t>(width));
  }

  if (rank == owner2) {
    diff_t offset = static_cast<diff_t>((global_found - start_row)) * static_cast<diff_t>(width);
    row_f.assign(local_matrix.begin() + offset, local_matrix.begin() + offset + static_cast<diff_t>(width));
  }

  std::array<MPI_Request, 2> reqs{};

  if (rank == owner) {
    MPI_Isend(row_k.data(), width, MPI_DOUBLE, owner2, 100 + k, MPI_COMM_WORLD, reqs.data());
    MPI_Irecv(row_f.data(), width, MPI_DOUBLE, owner2, 200 + k, MPI_COMM_WORLD, reqs.data() + 1);
    MPI_Waitall(2, reqs.data(), MPI_STATUSES_IGNORE);

    diff_t offset = static_cast<diff_t>((k - start_row)) * static_cast<diff_t>(width);
    std::ranges::copy(row_f, local_matrix.begin() + offset);

  } else if (rank == owner2) {
    MPI_Irecv(row_k.data(), width, MPI_DOUBLE, owner, 100 + k, MPI_COMM_WORLD, reqs.data());
    MPI_Isend(row_f.data(), width, MPI_DOUBLE, owner, 200 + k, MPI_COMM_WORLD, reqs.data() + 1);
    MPI_Waitall(2, reqs.data(), MPI_STATUSES_IGNORE);

    diff_t offset = static_cast<diff_t>((global_found - start_row)) * static_cast<diff_t>(width);
    std::ranges::copy(row_k, local_matrix.begin() + offset);
  }
}

void NormalizePivotRow(int rank, int owner, int k, int start_row, std::vector<double>& local_matrix,
                       std::vector<double>& pivot, int width, double k_eps) {
  if (rank == owner) {
    int lk = k - start_row;
    double piv = local_matrix[static_cast<std::size_t>((lk * width) + k)];
    if (std::fabs(piv) < k_eps) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int j = 0; j < width; ++j) {
      pivot[j] = local_matrix[static_cast<std::size_t>((lk * width) + j)] / piv;
    }
    std::ranges::copy(pivot, local_matrix.begin() + static_cast<std::vector<double>::difference_type>(lk * width));
  }

  MPI_Bcast(pivot.data(), width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
}

void EliminateColumn(int k, int start_row, int local_rows, std::vector<double>& local_matrix,
                     const std::vector<double>& pivot, int width) {
  for (int i = 0; i < local_rows; ++i) {
    int gi = start_row + i;
    if (gi == k) {
      continue;
    }
    double f = local_matrix[static_cast<std::size_t>((i * width) + k)];
    for (int j = 0; j < width; ++j) {
      local_matrix[static_cast<std::size_t>((i * width) + j)] -= f * pivot[j];
    }
  }
}

std::vector<double> ExtractLocalSolution(int local_rows, const std::vector<double>& local_matrix, int width, int n) {
  std::vector<double> local_solution(local_rows);
  for (int i = 0; i < local_rows; ++i) {
    local_solution[i] = local_matrix[static_cast<std::size_t>((i * width) + n)];
  }
  return local_solution;
}

std::vector<double> GatherSolutions(int rank, int size, const std::vector<int>& rows,
                                    const std::vector<double>& local_solution, int n) {
  std::vector<int> sol_counts(size);
  std::vector<int> sol_displs(size);

  int off = 0;
  for (int proc = 0; proc < size; ++proc) {
    sol_counts[proc] = rows[proc];
    sol_displs[proc] = off;
    off += sol_counts[proc];
  }

  std::vector<double> solution;
  if (rank == 0) {
    solution.resize(static_cast<std::size_t>(n));
  }

  MPI_Gatherv(local_solution.data(), static_cast<int>(local_solution.size()), MPI_DOUBLE,
              rank == 0 ? solution.data() : nullptr, sol_counts.data(), sol_displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  return solution;
}

bool SetFinalOutput(int rank, const std::vector<double>& solution, std::vector<float>& output_ref) {
  if (rank == 0) {
    std::vector<float> final_solution(solution.size());
    for (std::size_t i = 0; i < solution.size(); ++i) {
      final_solution[i] = static_cast<float>(solution[i]);
    }
    output_ref = final_solution;
  }
  return true;
}

}  // namespace

bool ZyazevaSGaussJordanElMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto& input = GetInput();

  // Встроенный функционал GetMatrixSize
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(input.size());
  }
  static_cast<void>(input);  // Показываем, что параметр используется
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int width = n + 1;

  // Встроенный функционал PrepareFlatMatrix
  std::vector<double> flat(static_cast<std::size_t>(n) * static_cast<std::size_t>(width));

  if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < width; ++j) {
        flat[static_cast<std::size_t>(i * width + j)] =
            static_cast<double>(input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
      }
    }
  }

  MPI_Bcast(flat.data(), n * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> rows = DistributeRows(size, n);

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  CalculateScatterParameters(size, rows, width, send_counts, displs);

  int local_rows = rows[rank];
  std::vector<double> local_matrix(static_cast<std::size_t>(local_rows) * width);

  ScatterMatrix(rank, flat, send_counts, displs, local_matrix, local_rows, width);

  int start_row = CalculateStartRow(rank, rows);

  const double k_eps = 1e-12;
  std::vector<double> pivot(width);

  // Основной цикл Гаусса-Жордана
  for (int k = 0; k < n; ++k) {
    int owner = FindRowOwner(k, size, rows);

    int local_found = FindLocalPivotRow(k, start_row, local_rows, local_matrix, width, k_eps);
    int global_found = FindGlobalPivotRow(local_found);

    if (global_found == INT_MAX) {
      HandleZeroPivot(rank, owner, pivot, width);
      continue;
    }

    int owner2 = FindRowOwner(global_found, size, rows);

    if (owner == owner2) {
      SwapRowsSameOwner(rank, owner, k, global_found, start_row, local_matrix, width);
    } else {
      SwapRowsDifferentOwners(rank, owner, owner2, k, global_found, start_row, local_matrix, width);
    }

    NormalizePivotRow(rank, owner, k, start_row, local_matrix, pivot, width, k_eps);
    EliminateColumn(k, start_row, local_rows, local_matrix, pivot, width);
  }

  std::vector<double> local_solution = ExtractLocalSolution(local_rows, local_matrix, width, n);
  std::vector<double> solution = GatherSolutions(rank, size, rows, local_solution, n);

  auto& output_ref = const_cast<std::vector<float>&>(GetOutput());
  return SetFinalOutput(rank, solution, output_ref);
}

bool ZyazevaSGaussJordanElMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return !GetOutput().empty();
  }
  return true;
}

}  // namespace zyazeva_s_gauss_jordan_elimination

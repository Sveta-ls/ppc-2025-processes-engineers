#include "zyazeva_s_gauss_jordan_elimination/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <cmath>
#include "util/include/util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

ZyazevaSGaussJordanElMPI::ZyazevaSGaussJordanElMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZyazevaSGaussJordanElMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;

  if (rank == 0) {
    const auto& matrix = GetInput();

    if (matrix.empty()) {
      is_valid = false;
    } else {
      int n = matrix.size();
      is_valid = true;

      for (const auto& row : matrix) {
        if (row.size() != static_cast<size_t>(n + 1)) {
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

bool ZyazevaSGaussJordanElMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto& input = GetInput();
  int n = (rank == 0 ? static_cast<int>(input.size()) : 0);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    if (rank == 0) {
      GetOutput() = std::vector<float>();
    }
    return true;
  }
  const int width = n + 1;

  std::vector<double> flat;
  if (rank == 0) {
    flat.resize(static_cast<size_t>(n) * width);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < width; ++j) {
        flat[i * width + j] = static_cast<double>(input[i][j]);
      }
    }
  }

  std::vector<int> rows(size);
  int base = n / size;
  int extra = n % size;
  for (int p = 0; p < size; ++p) {
    rows[p] = base + (p < extra ? 1 : 0);
  }

  std::vector<int> send_counts(size), displs(size);
  int offset_rows = 0;
  for (int p = 0; p < size; ++p) {
    send_counts[p] = rows[p] * width;
    displs[p] = offset_rows * width;
    offset_rows += rows[p];
  }

  int local_rows = rows[rank];
  std::vector<double> local_matrix(local_rows * width);

  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, send_counts.data(), displs.data(), MPI_DOUBLE, local_matrix.data(),
               local_rows * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start_row = 0;
  for (int p = 0; p < rank; ++p) {
    start_row += rows[p];
  }

  std::vector<double> pivot(width);
  const double EPS = 1e-12;

  for (int k = 0; k < n; ++k) {
    int owner = 0, acc = 0;
    for (int p = 0; p < size; ++p) {
      if (k >= acc && k < acc + rows[p]) {
        owner = p;
        break;
      }
      acc += rows[p];
    }

    int local_found = INT_MAX;
    for (int i = 0; i < local_rows; ++i) {
      int gi = start_row + i;
      if (gi < k) {
        continue;
      }
      if (std::fabs(local_matrix[i * width + k]) > EPS) {
        local_found = gi;
        break;
      }
    }

    int global_found;
    MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (global_found == INT_MAX) {
      if (rank == owner) {
        std::fill(pivot.begin(), pivot.end(), 0.0);
      }
      MPI_Bcast(pivot.data(), width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
      continue;
    }

    int owner2 = 0, acc2 = 0;
    for (int p = 0; p < size; ++p) {
      if (global_found >= acc2 && global_found < acc2 + rows[p]) {
        owner2 = p;
        break;
      }
      acc2 += rows[p];
    }

    if (owner == owner2) {
      if (rank == owner) {
        int lk = k - start_row;
        int lf = global_found - start_row;
        for (int j = 0; j < width; ++j) {
          std::swap(local_matrix[lk * width + j], local_matrix[lf * width + j]);
        }
      }
    } else {
      std::vector<double> row_k, row_f;
      if (rank == owner) {
        row_k.assign(local_matrix.begin() + (k - start_row) * width,
                     local_matrix.begin() + (k - start_row + 1) * width);
      }
      if (rank == owner2) {
        row_f.assign(local_matrix.begin() + (global_found - start_row) * width,
                     local_matrix.begin() + (global_found - start_row + 1) * width);
      }

      MPI_Request reqs[2];
      if (rank == owner) {
        MPI_Isend(row_k.data(), width, MPI_DOUBLE, owner2, 100 + k, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(row_f.data(), width, MPI_DOUBLE, owner2, 200 + k, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        std::copy(row_f.begin(), row_f.end(), local_matrix.begin() + (k - start_row) * width);
      } else if (rank == owner2) {
        MPI_Irecv(row_k.data(), width, MPI_DOUBLE, owner, 100 + k, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(row_f.data(), width, MPI_DOUBLE, owner, 200 + k, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        std::copy(row_k.begin(), row_k.end(), local_matrix.begin() + (global_found - start_row) * width);
      }
    }

    if (rank == owner) {
      int lk = k - start_row;
      double piv = local_matrix[lk * width + k];
      if (std::fabs(piv) < EPS) {
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      for (int j = 0; j < width; ++j) {
        pivot[j] = local_matrix[lk * width + j] / piv;
      }
      std::copy(pivot.begin(), pivot.end(), local_matrix.begin() + lk * width);
    }

    MPI_Bcast(pivot.data(), width, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
      int gi = start_row + i;
      if (gi == k) {
        continue;
      }
      double f = local_matrix[i * width + k];
      for (int j = 0; j < width; ++j) {
        local_matrix[i * width + j] -= f * pivot[j];
      }
    }
  }

  std::vector<double> local_solution(local_rows);
  for (int i = 0; i < local_rows; ++i) {
    local_solution[i] = local_matrix[i * width + n];
  }

  std::vector<int> sol_counts(size), sol_displs(size);
  int off = 0;
  for (int p = 0; p < size; ++p) {
    sol_counts[p] = rows[p];
    sol_displs[p] = off;
    off += sol_counts[p];
  }

  std::vector<double> solution;
  if (rank == 0) {
    solution.resize(n);
  }

  MPI_Gatherv(local_solution.data(), local_rows, MPI_DOUBLE, rank == 0 ? solution.data() : nullptr, sol_counts.data(),
              sol_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<float> final_solution(n);
    for (int i = 0; i < n; ++i) {
      final_solution[i] = static_cast<float>(solution[i]);
    }
    GetOutput() = final_solution;
  }

  return true;
}

bool ZyazevaSGaussJordanElMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return !GetOutput().empty();
  }
  return true;
}

}  // namespace zyazeva_s_gauss_jordan_elimination

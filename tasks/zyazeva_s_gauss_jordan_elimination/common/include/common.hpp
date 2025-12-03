#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyazeva_s_gauss_jordan_elimination

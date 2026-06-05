#pragma once

#include "src/types.h"

namespace top_k {

DenseMatrix compute_eigen_direct_katz_matrix(const Graph &graph, double alpha);

}  // namespace top_k

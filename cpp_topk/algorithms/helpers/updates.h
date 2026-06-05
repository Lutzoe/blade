#pragma once

#include "src/types.h"

namespace top_k {

bool sherman_morrison_update(
    DenseMatrix &katz,
    std::vector<double> &scores,
    std::size_t u,
    std::size_t v,
    double alpha
);

std::vector<double> jacobi_scores(
    const Graph &graph,
    double alpha,
    const std::vector<double> &warm_start
);

bool incremental_katz_scores_after_edges(
    const Graph &graph,
    double alpha,
    const std::vector<Edge> &added_edges,
    std::vector<double> &scores
);

bool apply_update_mode(
    const UpdateModeSelection &selection,
    DenseMatrix &katz,
    std::vector<double> &scores,
    const Graph &graph,
    std::size_t u,
    std::size_t v,
    double alpha
);

}  // namespace top_k

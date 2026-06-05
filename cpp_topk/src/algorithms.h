#pragma once

#include "src/types.h"

namespace top_k {

AlgorithmResult run_gap_greedy_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_katz_mass_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_optimal_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_dense_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_blade_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_blade_no_batch_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

AlgorithmResult run_same_group_support_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);

}  // namespace top_k

#pragma once

#include "src/types.h"

#include <optional>
#include <random>
#include <vector>

namespace top_k {

std::optional<std::string> underrepresented_group(
    const std::map<std::string, int> &counts,
    const std::map<std::string, int> &target_counts
);

std::vector<std::size_t> vulnerable_nodes(
    const Graph &graph,
    const std::vector<std::size_t> &top_k,
    const std::optional<std::string> &group
);

bool has_two_hop_addable_source_to_target(
    const Graph &graph,
    std::size_t target
);

bool is_two_hop_addable_edge(
    const Graph &graph,
    std::size_t source,
    std::size_t target
);

std::vector<char> two_hop_addable_sources_to_target(
    const Graph &graph,
    std::size_t target
);

bool is_addable_edge(
    const Graph &graph,
    std::size_t source,
    std::size_t target,
    const std::string &edge_admissibility
);

bool has_addable_source_to_target(
    const Graph &graph,
    std::size_t target,
    const std::string &edge_admissibility
);

std::vector<char> addable_sources_to_target(
    const Graph &graph,
    std::size_t target,
    const std::string &edge_admissibility
);

std::vector<Edge> candidate_edges(
    const Graph &graph,
    const std::optional<std::vector<std::size_t>> &sources = std::nullopt,
    const std::optional<std::vector<std::size_t>> &targets = std::nullopt,
    const std::string &edge_admissibility = "any"
);

std::vector<Edge> random_candidate_edges(
    const Graph &graph,
    std::size_t sample_size,
    std::mt19937 &rng,
    const std::optional<std::vector<std::size_t>> &sources = std::nullopt,
    const std::optional<std::vector<std::size_t>> &targets = std::nullopt,
    const std::string &edge_admissibility = "any"
);

GapTraceEntry snapshot(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts,
    int step,
    const std::optional<Edge> &edge
);

double differential_gain(
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    double alpha,
    std::size_t a,
    std::size_t b,
    std::size_t u,
    std::size_t v
);

}  // namespace top_k

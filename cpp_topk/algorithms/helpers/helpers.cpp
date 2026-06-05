#include "algorithms/helpers/helpers.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <unordered_set>

namespace top_k {

std::optional<std::string> underrepresented_group(
    const std::map<std::string, int> &counts,
    const std::map<std::string, int> &target_counts
) {
    std::set<std::string> keys;
    for (const auto &[key, _] : counts) {
        keys.insert(key);
    }
    for (const auto &[key, _] : target_counts) {
        keys.insert(key);
    }
    std::optional<std::string> best;
    int best_deficit = 0;
    for (const auto &key : keys) {
        const int have = counts.count(key) ? counts.at(key) : 0;
        const int want = target_counts.count(key) ? target_counts.at(key) : 0;
        const int deficit = want - have;
        if (deficit > best_deficit) {
            best = key;
            best_deficit = deficit;
        }
    }
    return best;
}

std::vector<std::size_t> vulnerable_nodes(
    const Graph &graph,
    const std::vector<std::size_t> &top_k,
    const std::optional<std::string> &group
) {
    std::vector<std::size_t> result;
    for (std::size_t node : top_k) {
        if (!group.has_value() || graph.groups[node] != *group) {
            result.push_back(node);
        }
    }
    return result;
}

bool has_two_hop_addable_source_to_target(
    const Graph &graph,
    std::size_t target
) {
    if (target >= graph.nodes.size()) {
        return false;
    }
    for (std::size_t middle : graph.in_neighbors[target]) {
        for (std::size_t source : graph.in_neighbors[middle]) {
            if (source != target && !graph.has_edge(source, target)) {
                return true;
            }
        }
    }
    return false;
}

bool is_two_hop_addable_edge(
    const Graph &graph,
    std::size_t source,
    std::size_t target
) {
    if (source >= graph.nodes.size() || target >= graph.nodes.size() ||
        source == target || graph.has_edge(source, target)) {
        return false;
    }
    const auto &source_out = graph.out_neighbors[source];
    const auto &target_in = graph.in_neighbors[target];
    auto out_it = source_out.begin();
    auto in_it = target_in.begin();
    while (out_it != source_out.end() && in_it != target_in.end()) {
        if (*out_it == *in_it) {
            return true;
        }
        if (*out_it < *in_it) {
            ++out_it;
        } else {
            ++in_it;
        }
    }
    return false;
}

std::vector<char> two_hop_addable_sources_to_target(
    const Graph &graph,
    std::size_t target
) {
    const std::size_t n = graph.nodes.size();
    std::vector<char> addable(n, 0);
    if (target >= n) {
        return addable;
    }

    for (std::size_t middle : graph.in_neighbors[target]) {
        for (std::size_t source : graph.in_neighbors[middle]) {
            addable[source] = 1;
        }
    }

    addable[target] = 0;
    for (std::size_t source : graph.in_neighbors[target]) {
        addable[source] = 0;
    }
    return addable;
}

bool is_addable_edge(
    const Graph &graph,
    std::size_t source,
    std::size_t target,
    const std::string &edge_admissibility
) {
    if (edge_admissibility == "two_hop") {
        return is_two_hop_addable_edge(graph, source, target);
    }
    return source < graph.nodes.size() &&
           target < graph.nodes.size() &&
           source != target &&
           !graph.has_edge(source, target);
}

bool has_addable_source_to_target(
    const Graph &graph,
    std::size_t target,
    const std::string &edge_admissibility
) {
    if (edge_admissibility == "two_hop") {
        return has_two_hop_addable_source_to_target(graph, target);
    }
    return target < graph.nodes.size() &&
           graph.in_neighbors[target].size() + 1 < graph.nodes.size();
}

std::vector<char> addable_sources_to_target(
    const Graph &graph,
    std::size_t target,
    const std::string &edge_admissibility
) {
    if (edge_admissibility == "two_hop") {
        return two_hop_addable_sources_to_target(graph, target);
    }

    const std::size_t n = graph.nodes.size();
    std::vector<char> addable(n, 0);
    if (target >= n) {
        return addable;
    }
    std::fill(addable.begin(), addable.end(), 1);
    addable[target] = 0;
    for (std::size_t source : graph.in_neighbors[target]) {
        if (source < addable.size()) {
            addable[source] = 0;
        }
    }
    return addable;
}

std::vector<Edge> candidate_edges(
    const Graph &graph,
    const std::optional<std::vector<std::size_t>> &sources,
    const std::optional<std::vector<std::size_t>> &targets,
    const std::string &edge_admissibility
) {
    const std::vector<std::size_t> source_nodes = sources.has_value()
        ? *sources
        : [&graph]() {
              std::vector<std::size_t> ids(graph.nodes.size());
              std::iota(ids.begin(), ids.end(), 0);
              return ids;
          }();
    const std::vector<std::size_t> target_nodes = targets.has_value()
        ? *targets
        : [&graph]() {
              std::vector<std::size_t> ids(graph.nodes.size());
              std::iota(ids.begin(), ids.end(), 0);
              return ids;
          }();

    std::vector<Edge> edges;
    for (std::size_t u : source_nodes) {
        for (std::size_t v : target_nodes) {
            if (!is_addable_edge(graph, u, v, edge_admissibility)) {
                continue;
            }
            edges.push_back(Edge{u, v});
        }
    }
    return edges;
}

std::vector<Edge> random_candidate_edges(
    const Graph &graph,
    std::size_t sample_size,
    std::mt19937 &rng,
    const std::optional<std::vector<std::size_t>> &sources,
    const std::optional<std::vector<std::size_t>> &targets,
    const std::string &edge_admissibility
) {
    if (sample_size == 0 || graph.nodes.empty()) {
        return {};
    }

    const std::vector<std::size_t> source_nodes = sources.has_value()
        ? *sources
        : [&graph]() {
              std::vector<std::size_t> ids(graph.nodes.size());
              std::iota(ids.begin(), ids.end(), 0);
              return ids;
          }();
    const std::vector<std::size_t> target_nodes = targets.has_value()
        ? *targets
        : [&graph]() {
              std::vector<std::size_t> ids(graph.nodes.size());
              std::iota(ids.begin(), ids.end(), 0);
              return ids;
          }();

    if (source_nodes.empty() || target_nodes.empty()) {
        return {};
    }

    const std::size_t total_pairs = source_nodes.size() * target_nodes.size();
    if (sample_size * 4 >= total_pairs) {
        std::vector<Edge> edges = candidate_edges(
            graph,
            source_nodes,
            target_nodes,
            edge_admissibility
        );
        std::shuffle(edges.begin(), edges.end(), rng);
        if (edges.size() > sample_size) {
            edges.resize(sample_size);
        }
        return edges;
    }

    std::vector<Edge> edges;
    edges.reserve(sample_size);
    std::unordered_set<std::size_t> seen_offsets;
    const std::size_t max_attempts = std::max<std::size_t>(sample_size * 100, 10000);
    std::uniform_int_distribution<std::size_t> source_dist(0, source_nodes.size() - 1);
    std::uniform_int_distribution<std::size_t> target_dist(0, target_nodes.size() - 1);
    for (std::size_t attempt = 0; edges.size() < sample_size && attempt < max_attempts; ++attempt) {
        const std::size_t source_offset = source_dist(rng);
        const std::size_t target_offset = target_dist(rng);
        const std::size_t offset = source_offset * target_nodes.size() + target_offset;
        if (!seen_offsets.insert(offset).second) {
            continue;
        }
        const std::size_t u = source_nodes[source_offset];
        const std::size_t v = target_nodes[target_offset];
        if (!is_addable_edge(graph, u, v, edge_admissibility)) {
            continue;
        }
        edges.push_back(Edge{u, v});
    }

    return edges;
}

GapTraceEntry snapshot(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts,
    int step,
    const std::optional<Edge> &edge
) {
    GapTraceEntry entry;
    entry.step = step;
    entry.edge = edge;
    entry.top_k_ranking = tie_aware_top_k_ranking(graph, scores, k, target_counts);
    entry.top_k = top_k_nodes(entry.top_k_ranking, k);
    entry.counts = group_counts(graph, entry.top_k);
    entry.gap = fairness_gap(entry.counts, target_counts);
    entry.katz_mass_objective = katz_mass_objective(graph, scores, target_counts);
    return entry;
}

double differential_gain(
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    double alpha,
    std::size_t a,
    std::size_t b,
    std::size_t u,
    std::size_t v
) {
    const double eta = 1.0 - alpha * katz.at(v, u);
    if (eta <= kEps) {
        return 0.0;
    }
    const double lambda = alpha * scores[u] / eta;
    const double mu = katz.at(v, a) - katz.at(v, b);
    return lambda * mu;
}

}  // namespace top_k

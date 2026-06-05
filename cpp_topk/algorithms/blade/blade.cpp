#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

namespace top_k {
namespace {

std::vector<std::size_t> ranked_sources(const std::vector<double> &scores) {
    std::vector<std::size_t> sources(scores.size());
    for (std::size_t node = 0; node < scores.size(); ++node) {
        sources[node] = node;
    }
    std::sort(sources.begin(), sources.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (std::abs(scores[lhs] - scores[rhs]) > kEps) {
            return scores[lhs] > scores[rhs];
        }
        return lhs < rhs;
    });
    return sources;
}

std::optional<std::size_t> lowest_score_node(
    const std::vector<std::size_t> &nodes,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k
) {
    std::vector<std::size_t> top_k_position(scores.size(), top_k.size());
    for (std::size_t i = 0; i < top_k.size(); ++i) {
        top_k_position[top_k[i]] = i;
    }
    std::optional<std::size_t> best;
    for (std::size_t node : nodes) {
        if (!best.has_value() || scores[node] < scores[*best] ||
            (std::abs(scores[node] - scores[*best]) <= kEps &&
             top_k_position[node] > top_k_position[*best])) {
            best = node;
        }
    }
    return best;
}

std::vector<std::size_t> top_promotable_nodes(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k,
    const std::string &group,
    int limit,
    const std::string &edge_admissibility
) {
    const std::size_t capped_limit = static_cast<std::size_t>(std::max(1, limit));
    std::vector<char> in_top_k(graph.nodes.size(), 0);
    for (std::size_t node : top_k) {
        in_top_k[node] = 1;
    }

    std::vector<std::size_t> result;
    result.reserve(capped_limit);
    const std::vector<RankedNode> ranking = rank_scores(scores);
    for (const RankedNode &item : ranking) {
        const std::size_t node = item.index;
        if (in_top_k[node] != 0 || graph.groups[node] != group) {
            continue;
        }
        if (!has_addable_source_to_target(graph, node, edge_admissibility)) {
            continue;
        }
        result.push_back(node);
        if (result.size() >= capped_limit) {
            break;
        }
    }
    return result;
}

struct EstimatedCandidate {
    std::size_t target = 0;
    double target_score = 0.0;
    std::vector<Edge> batch;
};

std::optional<EstimatedCandidate> estimate_candidate(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &source_order,
    std::size_t target,
    std::size_t boundary,
    double alpha,
    int budget_remainder,
    const std::string &edge_admissibility
) {
    EstimatedCandidate candidate;
    candidate.target = target;
    candidate.target_score = scores[target];
    candidate.batch.reserve(std::min<std::size_t>(
        source_order.size(),
        static_cast<std::size_t>(std::max(0, budget_remainder))
    ));

    double gap = std::max(kEps, scores[boundary] - scores[target] + kEps);
    const std::vector<char> addable_sources =
        addable_sources_to_target(graph, target, edge_admissibility);
    for (std::size_t source : source_order) {
        if (static_cast<int>(candidate.batch.size()) >= budget_remainder) {
            break;
        }
        if (source >= addable_sources.size() || addable_sources[source] == 0) {
            continue;
        }
        candidate.batch.push_back(Edge{source, target});
        gap -= alpha * std::max(0.0, scores[source]);
        if (gap <= 0.0) {
            return candidate;
        }
    }
    return std::nullopt;
}

bool better_candidate(const EstimatedCandidate &candidate, const EstimatedCandidate &best) {
    if (candidate.batch.size() != best.batch.size()) {
        return candidate.batch.size() < best.batch.size();
    }
    if (std::abs(candidate.target_score - best.target_score) > kEps) {
        return candidate.target_score > best.target_score;
    }
    return candidate.target < best.target;
}

int blade_budget_remainder(const ExperimentOptions &options, const Graph &graph) {
    if (options.budget >= 0) {
        return options.budget;
    }
    const std::size_t possible_edges = graph.nodes.size() > 1
        ? graph.nodes.size() * (graph.nodes.size() - 1)
        : 0;
    const std::size_t remaining_edges = possible_edges > graph.edge_count()
        ? possible_edges - graph.edge_count()
        : 0;
    return static_cast<int>(std::min<std::size_t>(
        remaining_edges,
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    ));
}

AlgorithmResult run_blade_impl(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options,
    bool commit_full_batch
) {
    AlgorithmResult result;
    result.algorithm_resolved = commit_full_batch ? "blade" : "blade_no_batch";
    const double target_tolerance = options.epsilon + kEps;
    result.update_mode_resolved = "jacobi_scores";
    result.gap_trace.push_back(snapshot(
        graph,
        initial.centrality.scores,
        options.k,
        options.target_counts,
        0,
        std::nullopt
    ));
    if (result.gap_trace.back().gap <= target_tolerance) {
        result.found = true;
        result.already_satisfied = true;
        result.final_scores = initial.centrality.scores;
        return result;
    }
    std::vector<double> scores = std::move(initial.centrality.scores);
    std::vector<std::size_t> top_k = std::move(initial.top_k);
    auto counts = initial.counts;
    const double alpha = initial.centrality.alpha;
    const int boundary_width = options.frontier_limit > 0
        ? options.frontier_limit
        : kDefaultBladeFrontierLimit;
    int budget_remainder = blade_budget_remainder(options, graph);
    std::string stop_reason = "loop condition ended.";

    while (budget_remainder > 0 && result.gap_trace.back().gap > target_tolerance) {
        const auto promoted_group = underrepresented_group(counts, options.target_counts);
        if (!promoted_group.has_value()) {
            stop_reason = "no underrepresented group remains.";
            break;
        }

        const auto promotable = top_promotable_nodes(
            graph,
            scores,
            top_k,
            *promoted_group,
            boundary_width,
            options.edge_admissibility
        );
        if (promotable.empty()) {
            stop_reason = "no promotable node exists.";
            break;
        }

        const auto opposing = vulnerable_nodes(graph, top_k, promoted_group);
        const auto boundary = lowest_score_node(opposing, scores, top_k);
        if (!boundary.has_value()) {
            stop_reason = "no opposing boundary node exists.";
            break;
        }

        const auto source_order = ranked_sources(scores);
        std::optional<EstimatedCandidate> best;
        for (std::size_t target : promotable) {
            const auto candidate = estimate_candidate(
                graph,
                scores,
                source_order,
                target,
                *boundary,
                alpha,
                budget_remainder,
                options.edge_admissibility
            );
            if (!candidate.has_value()) {
                continue;
            }
            if (!best.has_value() || better_candidate(*candidate, *best)) {
                best = *candidate;
            }
        }
        if (!best.has_value()) {
            stop_reason = "no finite-cost promoted candidate exists.";
            break;
        }

        bool added_any = false;
        std::optional<Edge> last_edge;
        const std::size_t commit_count = commit_full_batch
            ? best->batch.size()
            : std::min<std::size_t>(1, best->batch.size());
        for (std::size_t i = 0; i < commit_count; ++i) {
            const Edge &edge = best->batch[i];
            if (budget_remainder <= 0) {
                break;
            }
            if (!graph.add_edge(edge.u, edge.v)) {
                continue;
            }
            result.edges.push_back(edge);
            last_edge = edge;
            added_any = true;
            --budget_remainder;
        }
        if (!added_any) {
            stop_reason = "selected candidate batch added no edge.";
            break;
        }

        scores = jacobi_scores(graph, alpha, scores);
        result.gap_trace.push_back(snapshot(
            graph,
            scores,
            options.k,
            options.target_counts,
            static_cast<int>(result.edges.size()),
            last_edge
        ));
        top_k = result.gap_trace.back().top_k;
        counts = result.gap_trace.back().counts;
        if (result.gap_trace.back().gap <= target_tolerance) {
            result.found = true;
            result.final_scores = scores;
            return result;
        }
    }

    if (budget_remainder <= 0 && result.gap_trace.back().gap > target_tolerance) {
        stop_reason = "budget was exhausted.";
    }
    result.final_scores = scores;
    if (!result.found && !result.already_satisfied) {
        std::ostringstream message;
        message << (commit_full_batch ? "blade" : "blade_no_batch")
                << " stopped before reaching the target using "
                << options.edge_admissibility << " admissible edges: "
                << stop_reason;
        result.message = message.str();
    }
    return result;
}

}  // namespace

AlgorithmResult run_blade_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    return run_blade_impl(graph, std::move(initial), options, true);
}

AlgorithmResult run_blade_no_batch_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    return run_blade_impl(graph, std::move(initial), options, false);
}

}  // namespace top_k

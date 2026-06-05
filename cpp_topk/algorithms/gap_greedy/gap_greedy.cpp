#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace top_k {
namespace {

inline constexpr int kDefaultGapGreedyFrontierLimit = 2;

int gap_greedy_budget_remainder(const ExperimentOptions &options, const Graph &graph) {
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

bool score_node_better(
    const std::vector<double> &scores,
    std::size_t lhs,
    std::size_t rhs
) {
    if (std::abs(scores[lhs] - scores[rhs]) > kEps) {
        return scores[lhs] > scores[rhs];
    }
    return lhs < rhs;
}

std::optional<std::size_t> lowest_score_node(
    const std::vector<std::size_t> &nodes,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k
) {
    std::vector<std::size_t> top_k_position(scores.size(), top_k.size());
    for (std::size_t position = 0; position < top_k.size(); ++position) {
        top_k_position[top_k[position]] = position;
    }

    std::optional<std::size_t> best;
    for (std::size_t node : nodes) {
        if (!best.has_value() ||
            scores[node] < scores[*best] - kEps ||
            (std::abs(scores[node] - scores[*best]) <= kEps &&
             top_k_position[node] > top_k_position[*best])) {
            best = node;
        }
    }
    return best;
}

std::vector<std::size_t> top_boundary_candidates(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k,
    const std::string &group,
    int limit,
    const std::string &edge_admissibility
) {
    std::vector<char> in_top_k(graph.nodes.size(), 0);
    for (std::size_t node : top_k) {
        in_top_k[node] = 1;
    }

    std::vector<std::size_t> result;
    result.reserve(static_cast<std::size_t>(std::max(1, limit)));
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
        if (static_cast<int>(result.size()) >= limit) {
            break;
        }
    }
    return result;
}

std::vector<std::size_t> ranked_sources(const std::vector<double> &scores) {
    std::vector<std::size_t> sources(scores.size());
    for (std::size_t node = 0; node < scores.size(); ++node) {
        sources[node] = node;
    }
    std::sort(sources.begin(), sources.end(), [&](std::size_t lhs, std::size_t rhs) {
        return score_node_better(scores, lhs, rhs);
    });
    return sources;
}

struct CandidateEdge {
    Edge edge;
    std::size_t target = 0;
    double old_boundary_gap = 0.0;
    double new_boundary_gap = 0.0;
    double boundary_gap_reduction = -std::numeric_limits<double>::infinity();
    double top_k_gap = std::numeric_limits<double>::infinity();
    std::vector<double> scores;
};

bool edge_less(const Edge &lhs, const Edge &rhs) {
    return std::tie(lhs.u, lhs.v) < std::tie(rhs.u, rhs.v);
}

bool better_candidate(const CandidateEdge &candidate, const CandidateEdge &best) {
    if (candidate.boundary_gap_reduction > best.boundary_gap_reduction + kEps) {
        return true;
    }
    if (best.boundary_gap_reduction > candidate.boundary_gap_reduction + kEps) {
        return false;
    }
    if (candidate.top_k_gap < best.top_k_gap - kEps) {
        return true;
    }
    if (best.top_k_gap < candidate.top_k_gap - kEps) {
        return false;
    }
    if (candidate.old_boundary_gap < best.old_boundary_gap - kEps) {
        return true;
    }
    if (best.old_boundary_gap < candidate.old_boundary_gap - kEps) {
        return false;
    }
    return edge_less(candidate.edge, best.edge);
}

std::optional<CandidateEdge> evaluate_jacobi_candidate(
    Graph &graph,
    const std::vector<double> &scores,
    const ExperimentOptions &options,
    double alpha,
    std::size_t target,
    std::size_t boundary,
    const Edge &edge
) {
    if (!graph.add_edge(edge.u, edge.v)) {
        return std::nullopt;
    }
    std::vector<double> candidate_scores = jacobi_scores(graph, alpha, scores);
    const GapTraceEntry candidate_snapshot = snapshot(
        graph,
        candidate_scores,
        options.k,
        options.target_counts,
        0,
        edge
    );

    auto &out_neighbors = graph.out_neighbors[edge.u];
    const auto out_it = std::lower_bound(out_neighbors.begin(), out_neighbors.end(), edge.v);
    if (out_it != out_neighbors.end() && *out_it == edge.v) {
        out_neighbors.erase(out_it);
    }
    auto &in_neighbors = graph.in_neighbors[edge.v];
    const auto in_it = std::lower_bound(in_neighbors.begin(), in_neighbors.end(), edge.u);
    if (in_it != in_neighbors.end() && *in_it == edge.u) {
        in_neighbors.erase(in_it);
    }

    CandidateEdge candidate;
    candidate.edge = edge;
    candidate.target = target;
    candidate.old_boundary_gap = scores[boundary] - scores[target];
    candidate.new_boundary_gap = candidate_scores[boundary] - candidate_scores[target];
    candidate.boundary_gap_reduction =
        candidate.old_boundary_gap - candidate.new_boundary_gap;
    candidate.top_k_gap = candidate_snapshot.gap;
    candidate.scores = std::move(candidate_scores);
    return candidate;
}

std::optional<CandidateEdge> evaluate_sherman_morrison_candidate(
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    const Graph &graph,
    const ExperimentOptions &options,
    double alpha,
    std::size_t target,
    std::size_t boundary,
    const Edge &edge
) {
    if (katz.n == 0 || edge.u >= scores.size() || edge.v >= scores.size()) {
        return std::nullopt;
    }
    double eta = 1.0 - alpha * katz.at(edge.v, edge.u);
    if (eta <= kEps) {
        return std::nullopt;
    }
    if (1.0 - eta <= kEps) {
        eta = 1.0;
    }

    std::vector<double> candidate_scores = scores;
    const double factor = alpha * scores[edge.u] / eta;
    for (std::size_t node = 0; node < candidate_scores.size(); ++node) {
        candidate_scores[node] += factor * katz.at(edge.v, node);
    }

    CandidateEdge candidate;
    candidate.edge = edge;
    candidate.target = target;
    candidate.old_boundary_gap = scores[boundary] - scores[target];
    candidate.boundary_gap_reduction =
        differential_gain(katz, scores, alpha, target, boundary, edge.u, edge.v);
    candidate.new_boundary_gap =
        candidate.old_boundary_gap - candidate.boundary_gap_reduction;
    candidate.top_k_gap = snapshot(
        graph,
        candidate_scores,
        options.k,
        options.target_counts,
        0,
        edge
    ).gap;
    candidate.scores = std::move(candidate_scores);
    return candidate;
}

}  // namespace

AlgorithmResult run_gap_greedy_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    AlgorithmResult result;
    const UpdateModeSelection update_mode =
        resolve_update_mode(options.update_mode, FairAlgorithmKind::GapGreedy);
    result.algorithm_resolved = "GapGreedy";
    result.update_mode_resolved = update_mode.resolved;

    const double target_tolerance = options.epsilon + kEps;
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

    const int frontier_limit = options.gap_greedy_frontier_limit > 0
        ? options.gap_greedy_frontier_limit
        : (options.frontier_limit > 0
            ? options.frontier_limit
            : kDefaultGapGreedyFrontierLimit);
    const double alpha = initial.centrality.alpha;
    DenseMatrix katz = std::move(initial.centrality.katz);
    std::vector<double> scores = std::move(initial.centrality.scores);
    std::vector<std::size_t> top_k = std::move(initial.top_k);
    std::map<std::string, int> counts = result.gap_trace.back().counts;
    int budget_remainder = gap_greedy_budget_remainder(options, graph);
    int recomputations = 0;
    std::string stop_reason = "loop condition ended.";

    while (budget_remainder > 0 && result.gap_trace.back().gap > target_tolerance) {
        const auto promoted_group =
            underrepresented_group(counts, options.target_counts);
        if (!promoted_group.has_value()) {
            stop_reason = "no underrepresented group remains.";
            break;
        }

        const std::vector<std::size_t> promotable = top_boundary_candidates(
            graph,
            scores,
            top_k,
            *promoted_group,
            frontier_limit,
            options.edge_admissibility
        );
        if (promotable.empty()) {
            stop_reason = "no promoted-group boundary candidate exists.";
            break;
        }

        const std::vector<std::size_t> opposing =
            vulnerable_nodes(graph, top_k, promoted_group);
        const auto boundary = lowest_score_node(opposing, scores, top_k);
        if (!boundary.has_value()) {
            stop_reason = "no opposing boundary node exists.";
            break;
        }

        const std::vector<std::size_t> source_order = ranked_sources(scores);
        std::optional<CandidateEdge> best;
        for (std::size_t target : promotable) {
            const std::vector<char> addable_sources =
                addable_sources_to_target(graph, target, options.edge_admissibility);
            for (std::size_t source : source_order) {
                if (source >= addable_sources.size() || addable_sources[source] == 0) {
                    continue;
                }
                const Edge edge{source, target};
                ++result.candidate_attempts;

                std::optional<CandidateEdge> candidate;
                if (update_mode.kind == UpdateModeKind::ShermanMorrison) {
                    candidate = evaluate_sherman_morrison_candidate(
                        katz,
                        scores,
                        graph,
                        options,
                        alpha,
                        target,
                        *boundary,
                        edge
                    );
                } else {
                    candidate = evaluate_jacobi_candidate(
                        graph,
                        scores,
                        options,
                        alpha,
                        target,
                        *boundary,
                        edge
                    );
                    ++recomputations;
                }

                if (!candidate.has_value() ||
                    candidate->boundary_gap_reduction <= kEps) {
                    continue;
                }
                if (!best.has_value() || better_candidate(*candidate, *best)) {
                    best = std::move(candidate);
                }
            }
        }

        if (!best.has_value()) {
            stop_reason = "no admissible edge had positive boundary-gap reduction.";
            break;
        }
        if (update_mode.kind == UpdateModeKind::ShermanMorrison) {
            if (!sherman_morrison_update(katz, scores, best->edge.u, best->edge.v, alpha)) {
                stop_reason = "selected edge failed Sherman-Morrison update.";
                break;
            }
        }
        if (!graph.add_edge(best->edge.u, best->edge.v)) {
            stop_reason = "selected edge could not be added.";
            break;
        }

        result.edges.push_back(best->edge);
        --budget_remainder;
        if (update_mode.kind != UpdateModeKind::ShermanMorrison) {
            scores = std::move(best->scores);
        }

        GapTraceEntry trace_entry = snapshot(
            graph,
            scores,
            options.k,
            options.target_counts,
            static_cast<int>(result.edges.size()),
            best->edge
        );
        top_k = trace_entry.top_k;
        counts = trace_entry.counts;
        result.gap_trace.push_back(std::move(trace_entry));
        if (result.gap_trace.back().gap <= target_tolerance) {
            result.found = true;
            stop_reason = "target tolerance was reached.";
            break;
        }
    }

    if (budget_remainder <= 0 && result.gap_trace.back().gap > target_tolerance) {
        stop_reason = "budget was exhausted.";
    }

    result.final_scores = std::move(scores);
    std::ostringstream message;
    message << "GapGreedy evaluated " << options.edge_admissibility
            << " admissible single incoming edges into the top "
            << frontier_limit
            << " outside-top-k promoted-group boundary candidates, selected the edge with largest positive exact one-edge reduction of the boundary gap against the current lowest-scoring opposing top-k node, and recomputed rankings after each committed edge; "
            << "final gap=" << result.gap_trace.back().gap
            << ", edges=" << result.edges.size()
            << ", candidate_attempts=" << result.candidate_attempts
            << ", jacobi_recomputations=" << recomputations
            << ", stop_reason=" << stop_reason << ".";
    result.message = message.str();
    return result;
}

}  // namespace top_k

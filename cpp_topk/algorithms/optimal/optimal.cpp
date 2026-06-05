#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace top_k {
namespace {

struct SearchState {
    std::vector<double> scores;
    double gap = 0.0;
    std::vector<Edge> edges;
    GapTraceEntry snapshot;
};

bool better_state(
    const SearchState &candidate,
    const std::optional<SearchState> &best
) {
    if (!best.has_value()) {
        return true;
    }
    if (candidate.gap != best->gap) {
        return candidate.gap < best->gap;
    }
    return std::lexicographical_compare(
        candidate.edges.begin(),
        candidate.edges.end(),
        best->edges.begin(),
        best->edges.end(),
        [](const Edge &lhs, const Edge &rhs) {
            if (lhs.u != rhs.u) {
                return lhs.u < rhs.u;
            }
            return lhs.v < rhs.v;
        }
    );
}

std::optional<SearchState> evaluate_edge_combination(
    const Graph &graph,
    const DenseMatrix &initial_katz,
    const std::vector<double> &initial_scores,
    const std::vector<Edge> &all_edges,
    const std::vector<std::size_t> &indices,
    const UpdateModeSelection &update_mode,
    const ExperimentOptions &options,
    double alpha
) {
    Graph candidate_graph = graph;
    std::vector<double> scores = initial_scores;
    std::vector<Edge> selected_edges;
    selected_edges.reserve(indices.size());
    for (std::size_t index : indices) {
        selected_edges.push_back(all_edges[index]);
    }

    if (update_mode.kind == UpdateModeKind::JacobiScores) {
        for (const Edge &edge : selected_edges) {
            if (!candidate_graph.add_edge(edge.u, edge.v)) {
                return std::nullopt;
            }
        }
        scores = jacobi_scores(candidate_graph, alpha, initial_scores);
    } else {
        DenseMatrix katz = initial_katz;
        for (const Edge &edge : selected_edges) {
            if (!apply_update_mode(
                update_mode,
                katz,
                scores,
                candidate_graph,
                edge.u,
                edge.v,
                alpha
            )) {
                return std::nullopt;
            }
            if (!candidate_graph.add_edge(edge.u, edge.v)) {
                return std::nullopt;
            }
        }
    }

    std::optional<Edge> last_edge;
    if (!selected_edges.empty()) {
        last_edge = selected_edges.back();
    }
    GapTraceEntry final_snapshot = snapshot(
        candidate_graph,
        scores,
        options.k,
        options.target_counts,
        static_cast<int>(selected_edges.size()),
        last_edge
    );

    SearchState state;
    state.scores = std::move(scores);
    state.gap = final_snapshot.gap;
    state.edges = std::move(selected_edges);
    state.snapshot = std::move(final_snapshot);
    return state;
}

std::optional<SearchState> search_exact_depth(
    const Graph &graph,
    const DenseMatrix &initial_katz,
    const std::vector<double> &initial_scores,
    const std::vector<Edge> &all_edges,
    const UpdateModeSelection &update_mode,
    const ExperimentOptions &options,
    double alpha,
    int depth,
    std::size_t &candidate_attempts
) {
    std::optional<SearchState> best;
    const std::size_t combination_size = static_cast<std::size_t>(depth);
    if (depth <= 0 || combination_size > all_edges.size()) {
        return best;
    }

    std::vector<std::size_t> indices(combination_size);
    std::iota(indices.begin(), indices.end(), 0);
    while (true) {
        ++candidate_attempts;
        auto candidate = evaluate_edge_combination(
            graph,
            initial_katz,
            initial_scores,
            all_edges,
            indices,
            update_mode,
            options,
            alpha
        );
        if (candidate.has_value()) {
            if (better_state(*candidate, best)) {
                best = std::move(candidate);
            }
            if (best.has_value() && best->gap <= kEps) {
                return best;
            }
        }

        std::size_t pivot = combination_size;
        while (pivot > 0 &&
               indices[pivot - 1] == all_edges.size() - combination_size + pivot - 1) {
            --pivot;
        }
        if (pivot == 0) {
            break;
        }
        ++indices[pivot - 1];
        for (std::size_t i = pivot; i < combination_size; ++i) {
            indices[i] = indices[i - 1] + 1;
        }
    }
    return best;
}

std::vector<Edge> sorted_candidate_edges(
    const Graph &graph,
    const std::string &edge_admissibility
) {
    std::vector<Edge> edges = candidate_edges(
        graph,
        std::nullopt,
        std::nullopt,
        edge_admissibility
    );
    std::sort(edges.begin(), edges.end(), [](const Edge &lhs, const Edge &rhs) {
        if (lhs.u != rhs.u) {
            return lhs.u < rhs.u;
        }
        return lhs.v < rhs.v;
    });
    return edges;
}

}  // namespace

AlgorithmResult run_optimal_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    if (graph.nodes.size() > 400) {
        throw std::runtime_error("optimal is limited to graphs with at most 400 nodes.");
    }

    AlgorithmResult result;
    const AlgorithmSelection algorithm_selection = resolve_algorithm_selection(options.algorithm);
    const UpdateModeSelection update_mode =
        resolve_update_mode(options.update_mode, algorithm_selection.kind);
    result.update_mode_resolved = update_mode.resolved;
    result.gap_trace.push_back(snapshot(
        graph,
        initial.centrality.scores,
        options.k,
        options.target_counts,
        0,
        std::nullopt
    ));
    if (result.gap_trace.back().gap <= kEps) {
        result.found = true;
        result.already_satisfied = true;
        result.final_scores = initial.centrality.scores;
        return result;
    }

    DenseMatrix initial_katz = std::move(initial.centrality.katz);
    std::vector<double> initial_scores = std::move(initial.centrality.scores);
    const std::vector<Edge> all_edges = sorted_candidate_edges(
        graph,
        options.edge_admissibility
    );

    std::optional<SearchState> chosen;
    const int max_depth = options.budget < 0
        ? static_cast<int>(all_edges.size())
        : options.budget;
    for (int depth = 1; depth <= max_depth; ++depth) {
        auto depth_best = search_exact_depth(
            graph,
            initial_katz,
            initial_scores,
            all_edges,
            update_mode,
            options,
            initial.centrality.alpha,
            depth,
            result.candidate_attempts
        );
        if (depth_best.has_value()) {
            if (!chosen.has_value() || better_state(*depth_best, chosen)) {
                chosen = std::move(depth_best);
            }
        }
        if (chosen.has_value() && chosen->gap <= kEps) {
            break;
        }
    }

    if (chosen.has_value()) {
        result.edges = std::move(chosen->edges);
        result.final_scores = std::move(chosen->scores);
        result.gap_trace.push_back(std::move(chosen->snapshot));
        result.found = result.gap_trace.back().gap <= kEps;
    }

    std::ostringstream message;
    message << "optimal brute-forced complete edge combinations up to "
            << (options.budget < 0 ? "all depths up to " : "depth ")
            << max_depth << " over " << all_edges.size()
            << " " << options.edge_admissibility
            << " addable edges, evaluating Katz after each complete combination with "
            << update_mode.resolved
            << ".";
    if (!result.found) {
        message << " No exact target-reaching combination was found within that depth.";
    }
    result.message = message.str();
    if (result.final_scores.empty()) {
        result.final_scores = initial_scores;
    }
    return result;
}

}  // namespace top_k

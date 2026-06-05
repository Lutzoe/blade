#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace top_k {
namespace {

inline constexpr double kIncrementalKatzResidualTolerance = 1e-11;
inline constexpr std::size_t kIncrementalKatzMaxRelaxations = 50000000;

struct OverlayKatzWorkspace {
    std::vector<double> residual;
    std::vector<double> score_delta;
    std::vector<char> residual_tracked;
    std::vector<char> delta_tracked;
    std::vector<char> queued;
    std::vector<char> queued_tracked;
    std::vector<std::size_t> residual_nodes;
    std::vector<std::size_t> changed_nodes;
    std::vector<std::size_t> queued_nodes;
    std::deque<std::size_t> queue;

    void ensure(std::size_t n) {
        if (residual.size() == n) {
            return;
        }
        residual.assign(n, 0.0);
        score_delta.assign(n, 0.0);
        residual_tracked.assign(n, 0);
        delta_tracked.assign(n, 0);
        queued.assign(n, 0);
        queued_tracked.assign(n, 0);
        residual_nodes.clear();
        changed_nodes.clear();
        queued_nodes.clear();
        queue.clear();
    }

    void reset() {
        for (std::size_t node : residual_nodes) {
            residual[node] = 0.0;
            residual_tracked[node] = 0;
        }
        for (std::size_t node : changed_nodes) {
            score_delta[node] = 0.0;
            delta_tracked[node] = 0;
        }
        for (std::size_t node : queued_nodes) {
            queued[node] = 0;
            queued_tracked[node] = 0;
        }
        residual_nodes.clear();
        changed_nodes.clear();
        queued_nodes.clear();
        queue.clear();
    }
};

struct WindowCandidate {
    std::size_t target = 0;
    std::size_t window_rank = 0;
    std::vector<Edge> edges;
    std::vector<std::pair<std::size_t, double>> score_deltas;
    double gap = std::numeric_limits<double>::infinity();
};

int same_group_budget_remainder(const ExperimentOptions &options, const Graph &graph) {
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

bool score_desc(const std::vector<double> &scores, std::size_t lhs, std::size_t rhs) {
    if (std::abs(scores[lhs] - scores[rhs]) > kEps) {
        return scores[lhs] > scores[rhs];
    }
    return lhs < rhs;
}

std::vector<char> top_k_mask(std::size_t n, const std::vector<std::size_t> &top_k) {
    std::vector<char> in_top_k(n, 0);
    for (std::size_t node : top_k) {
        if (node < in_top_k.size()) {
            in_top_k[node] = 1;
        }
    }
    return in_top_k;
}

std::vector<std::size_t> same_group_window_nodes(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k,
    const std::string &group,
    int window_size
) {
    const std::size_t limit = static_cast<std::size_t>(std::max(window_size, 0));
    if (limit == 0 || graph.nodes.empty()) {
        return {};
    }

    const std::vector<char> in_top_k = top_k_mask(graph.nodes.size(), top_k);
    auto better = [&scores](std::size_t lhs, std::size_t rhs) {
        return score_desc(scores, lhs, rhs);
    };
    std::priority_queue<
        std::size_t,
        std::vector<std::size_t>,
        decltype(better)
    > heap(better);

    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        if (node >= graph.groups.size() || graph.groups[node] != group ||
            in_top_k[node] != 0) {
            continue;
        }
        if (heap.size() < limit) {
            heap.push(node);
            continue;
        }
        if (score_desc(scores, node, heap.top())) {
            heap.pop();
            heap.push(node);
        }
    }

    std::vector<std::size_t> window;
    window.reserve(heap.size());
    while (!heap.empty()) {
        window.push_back(heap.top());
        heap.pop();
    }
    std::sort(window.begin(), window.end(), [&scores](std::size_t lhs, std::size_t rhs) {
        return score_desc(scores, lhs, rhs);
    });
    return window;
}

std::vector<Edge> bidirectional_window_support_edges(
    const Graph &graph,
    const std::vector<std::size_t> &window,
    std::size_t target,
    int budget_remainder,
    const std::string &edge_admissibility
) {
    std::vector<Edge> edges;
    if (target >= graph.groups.size() || budget_remainder <= 0) {
        return edges;
    }
    const std::size_t reserve_size = std::min<std::size_t>(
        static_cast<std::size_t>(budget_remainder),
        window.empty() ? 0 : 2 * (window.size() - 1)
    );
    edges.reserve(reserve_size);

    const std::string &group = graph.groups[target];
    const std::vector<char> incoming_sources =
        addable_sources_to_target(graph, target, edge_admissibility);
    std::vector<char> outgoing_targets(graph.nodes.size(), 0);
    if (edge_admissibility == "two_hop") {
        for (std::size_t middle : graph.out_neighbors[target]) {
            if (middle >= graph.out_neighbors.size()) {
                continue;
            }
            for (std::size_t candidate_target : graph.out_neighbors[middle]) {
                outgoing_targets[candidate_target] = 1;
            }
        }
        outgoing_targets[target] = 0;
        for (std::size_t existing_target : graph.out_neighbors[target]) {
            outgoing_targets[existing_target] = 0;
        }
    } else {
        std::fill(outgoing_targets.begin(), outgoing_targets.end(), 1);
        outgoing_targets[target] = 0;
        for (std::size_t existing_target : graph.out_neighbors[target]) {
            if (existing_target < outgoing_targets.size()) {
                outgoing_targets[existing_target] = 0;
            }
        }
    }

    for (std::size_t source : window) {
        if (static_cast<int>(edges.size()) >= budget_remainder) {
            break;
        }
        if (source == target || source >= graph.groups.size() ||
            graph.groups[source] != group) {
            continue;
        }
        if (source < incoming_sources.size() && incoming_sources[source] != 0) {
            edges.push_back(Edge{source, target});
            if (static_cast<int>(edges.size()) >= budget_remainder) {
                break;
            }
        }
        if (source < outgoing_targets.size() && outgoing_targets[source] != 0) {
            edges.push_back(Edge{target, source});
        }
    }
    return edges;
}

bool incremental_katz_score_delta_after_edges(
    const Graph &graph,
    double alpha,
    const std::vector<Edge> &added_edges,
    const std::vector<double> &base_scores,
    OverlayKatzWorkspace &workspace
) {
    const std::size_t n = graph.nodes.size();
    workspace.ensure(n);
    workspace.reset();
    if (base_scores.size() != n || alpha <= 0.0 || added_edges.empty()) {
        return base_scores.size() == n;
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>> overlay_out;
    overlay_out.reserve(added_edges.size() * 2);
    auto enqueue = [&](std::size_t node) {
        if (node >= n || workspace.queued[node] != 0 ||
            std::abs(workspace.residual[node]) <= kIncrementalKatzResidualTolerance) {
            return;
        }
        workspace.queued[node] = 1;
        if (workspace.queued_tracked[node] == 0) {
            workspace.queued_tracked[node] = 1;
            workspace.queued_nodes.push_back(node);
        }
        workspace.queue.push_back(node);
    };
    auto add_residual = [&](std::size_t node, double value) {
        if (node >= n || std::abs(value) <= kIncrementalKatzResidualTolerance) {
            return;
        }
        if (workspace.residual_tracked[node] == 0) {
            workspace.residual_tracked[node] = 1;
            workspace.residual_nodes.push_back(node);
        }
        workspace.residual[node] += value;
        enqueue(node);
    };

    for (const Edge &edge : added_edges) {
        if (edge.u >= n || edge.v >= n) {
            workspace.reset();
            return false;
        }
        overlay_out[edge.u].push_back(edge.v);
        add_residual(edge.v, alpha * base_scores[edge.u]);
    }

    std::size_t relaxations = 0;
    while (!workspace.queue.empty()) {
        if (++relaxations > kIncrementalKatzMaxRelaxations) {
            workspace.reset();
            return false;
        }
        const std::size_t node = workspace.queue.front();
        workspace.queue.pop_front();
        workspace.queued[node] = 0;
        const double delta = workspace.residual[node];
        workspace.residual[node] = 0.0;
        if (std::abs(delta) <= kIncrementalKatzResidualTolerance) {
            continue;
        }
        if (workspace.delta_tracked[node] == 0) {
            workspace.delta_tracked[node] = 1;
            workspace.changed_nodes.push_back(node);
        }
        workspace.score_delta[node] += delta;

        const double propagated = alpha * delta;
        if (std::abs(propagated) <= kIncrementalKatzResidualTolerance) {
            continue;
        }
        for (std::size_t next : graph.out_neighbors[node]) {
            add_residual(next, propagated);
        }
        const auto overlay_it = overlay_out.find(node);
        if (overlay_it == overlay_out.end()) {
            continue;
        }
        for (std::size_t next : overlay_it->second) {
            add_residual(next, propagated);
        }
    }
    return true;
}

double candidate_gap_from_deltas(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k,
    const std::vector<char> &in_top_k,
    const OverlayKatzWorkspace &workspace,
    int k,
    const std::map<std::string, int> &target_counts
) {
    std::vector<std::size_t> pool;
    pool.reserve(top_k.size() + workspace.changed_nodes.size());
    for (std::size_t node : top_k) {
        if (node < scores.size()) {
            pool.push_back(node);
        }
    }
    for (std::size_t node : workspace.changed_nodes) {
        if (node < in_top_k.size() && in_top_k[node] == 0) {
            pool.push_back(node);
        }
    }

    auto score_with_delta = [&](std::size_t node) {
        return scores[node] + workspace.score_delta[node];
    };
    std::sort(pool.begin(), pool.end(), [&](std::size_t lhs, std::size_t rhs) {
        const double lhs_score = score_with_delta(lhs);
        const double rhs_score = score_with_delta(rhs);
        if (std::abs(lhs_score - rhs_score) > kEps) {
            return lhs_score > rhs_score;
        }
        return lhs < rhs;
    });
    if (pool.size() > static_cast<std::size_t>(std::max(k, 0))) {
        pool.resize(static_cast<std::size_t>(std::max(k, 0)));
    }
    return fairness_gap(group_counts(graph, pool), target_counts);
}

std::vector<std::pair<std::size_t, double>> score_deltas_from_workspace(
    const OverlayKatzWorkspace &workspace
) {
    std::vector<std::pair<std::size_t, double>> deltas;
    deltas.reserve(workspace.changed_nodes.size());
    for (std::size_t node : workspace.changed_nodes) {
        const double delta = workspace.score_delta[node];
        if (std::abs(delta) > kIncrementalKatzResidualTolerance) {
            deltas.push_back({node, delta});
        }
    }
    return deltas;
}

bool better_candidate(
    const WindowCandidate &candidate,
    const std::optional<WindowCandidate> &best
) {
    if (!best.has_value()) {
        return true;
    }
    if (candidate.gap < best->gap - kEps) {
        return true;
    }
    if (std::abs(candidate.gap - best->gap) <= kEps &&
        candidate.window_rank < best->window_rank) {
        return true;
    }
    return false;
}

std::optional<WindowCandidate> best_window_candidate(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::vector<std::size_t> &top_k,
    const std::vector<std::size_t> &window,
    int budget_remainder,
    const ExperimentOptions &options,
    double alpha,
    double current_gap,
    std::size_t &candidate_attempts
) {
    std::optional<WindowCandidate> best;
    OverlayKatzWorkspace workspace;
    const std::vector<char> in_top_k = top_k_mask(graph.nodes.size(), top_k);

    for (std::size_t window_rank = 0; window_rank < window.size(); ++window_rank) {
        const std::size_t target = window[window_rank];
        const std::vector<Edge> edges = bidirectional_window_support_edges(
            graph,
            window,
            target,
            budget_remainder,
            options.edge_admissibility
        );
        if (edges.empty()) {
            continue;
        }
        ++candidate_attempts;
        if (!incremental_katz_score_delta_after_edges(
                graph,
                alpha,
                edges,
                scores,
                workspace
            )) {
            workspace.reset();
            continue;
        }

        const double candidate_gap = candidate_gap_from_deltas(
            graph,
            scores,
            top_k,
            in_top_k,
            workspace,
            options.k,
            options.target_counts
        );
        if (candidate_gap <= current_gap + kEps) {
            WindowCandidate candidate;
            candidate.target = target;
            candidate.window_rank = window_rank;
            candidate.edges = edges;
            candidate.score_deltas = score_deltas_from_workspace(workspace);
            candidate.gap = candidate_gap;
            if (better_candidate(candidate, best)) {
                best = std::move(candidate);
            }
        }
        workspace.reset();
    }

    return best;
}

void apply_score_deltas(
    const std::vector<std::pair<std::size_t, double>> &score_deltas,
    std::vector<double> &scores
) {
    for (const auto &[node, delta] : score_deltas) {
        if (node < scores.size()) {
            scores[node] += delta;
        }
    }
}

double final_gap_or_inf(const AlgorithmResult &result) {
    return result.gap_trace.empty()
        ? std::numeric_limits<double>::infinity()
        : result.gap_trace.back().gap;
}

}  // namespace

AlgorithmResult run_same_group_support_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    const UpdateModeSelection update_mode =
        resolve_update_mode(options.update_mode, FairAlgorithmKind::SameGroupSupport);
    Graph working_graph = graph;
    AlgorithmResult result;
    const double target_tolerance = options.epsilon + kEps;
    result.algorithm_resolved = "same_group_support_window";
    result.update_mode_resolved = update_mode.resolved;

    result.gap_trace.push_back(snapshot(
        working_graph,
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
    std::vector<std::size_t> top_k = result.gap_trace.back().top_k;
    auto counts = result.gap_trace.back().counts;
    int budget_remainder = same_group_budget_remainder(options, working_graph);
    int batches_completed = 0;
    std::string stop_reason = "loop condition ended.";
    std::optional<GapTraceEntry> last_before_reaching_target;

    while (budget_remainder > 0) {
        if (result.gap_trace.back().gap <= target_tolerance) {
            result.found = true;
            stop_reason = "target tolerance was reached.";
            break;
        }

        const auto group = underrepresented_group(counts, options.target_counts);
        if (!group.has_value()) {
            result.found = true;
            stop_reason = "target tolerance was reached.";
            break;
        }

        const std::vector<std::size_t> window = same_group_window_nodes(
            working_graph,
            scores,
            top_k,
            *group,
            options.same_group_window
        );
        if (window.empty()) {
            stop_reason = "same-group window was empty.";
            break;
        }

        const GapTraceEntry before_batch = result.gap_trace.back();
        const std::optional<WindowCandidate> candidate = best_window_candidate(
            working_graph,
            scores,
            top_k,
            window,
            budget_remainder,
            options,
            initial.centrality.alpha,
            before_batch.gap,
            result.candidate_attempts
        );
        if (!candidate.has_value()) {
            stop_reason = "no non-worsening same-group window batch exists.";
            break;
        }

        std::vector<Edge> added_edges;
        added_edges.reserve(candidate->edges.size());
        for (const Edge &edge : candidate->edges) {
            if (budget_remainder <= 0) {
                break;
            }
            if (edge.u >= working_graph.groups.size() ||
                edge.v >= working_graph.groups.size() ||
                working_graph.groups[edge.u] != working_graph.groups[edge.v]) {
                continue;
            }
            if (!working_graph.add_edge(edge.u, edge.v)) {
                continue;
            }
            added_edges.push_back(edge);
            result.edges.push_back(edge);
            --budget_remainder;
        }
        if (added_edges.empty()) {
            stop_reason = "selected same-group window batch added no edge.";
            break;
        }

        if (added_edges.size() == candidate->edges.size()) {
            apply_score_deltas(candidate->score_deltas, scores);
        } else {
            std::vector<double> updated_scores = scores;
            if (!incremental_katz_scores_after_edges(
                    working_graph,
                    initial.centrality.alpha,
                    added_edges,
                    updated_scores
                )) {
                updated_scores = jacobi_scores(
                    working_graph,
                    initial.centrality.alpha,
                    scores
                );
            }
            scores = std::move(updated_scores);
        }

        ++batches_completed;
        result.gap_trace.push_back(snapshot(
            working_graph,
            scores,
            options.k,
            options.target_counts,
            static_cast<int>(result.edges.size()),
            added_edges.back()
        ));
        top_k = result.gap_trace.back().top_k;
        counts = result.gap_trace.back().counts;

        if (result.gap_trace.back().gap <= target_tolerance) {
            result.found = true;
            last_before_reaching_target = before_batch;
            stop_reason = "target tolerance was reached.";
            break;
        }
    }

    if (budget_remainder <= 0 && !result.found) {
        stop_reason = "budget was exhausted.";
    }

    result.final_scores = scores;
    std::ostringstream message;
    message << "same_group_support used a same-group window of "
            << options.same_group_window
            << " top-ranked outside-top-k underrepresented nodes, evaluated "
            << "bidirectional support batches within that window, and selected "
            << "the best non-worsening top-k gap batch"
            << "; final gap=" << final_gap_or_inf(result)
            << ", edges=" << result.edges.size()
            << ", batches=" << batches_completed
            << ", same_group_window=" << options.same_group_window
            << ", support_scope=within_window"
            << ", edge_direction=bidirectional"
            << ", edge_admissibility=" << options.edge_admissibility
            << ", selection_rule=best_non_worsening_gap"
            << ", stop_reason=" << stop_reason;
    if (last_before_reaching_target.has_value()) {
        message << ", pre_stop_edges=" << last_before_reaching_target->step
                << ", pre_stop_gap=" << last_before_reaching_target->gap
                << ", pre_stop_counts={";
        bool first = true;
        for (const auto &[group, count] : last_before_reaching_target->counts) {
            if (!first) {
                message << ", ";
            }
            first = false;
            message << group << ":" << count;
        }
        message << "}";
    }
    message << ".";
    result.message = message.str();
    return result;
}

}  // namespace top_k

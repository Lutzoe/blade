#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace top_k {
namespace {

struct FrontierPair {
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t order = 0;
};

struct PairScan {
    bool finite = false;
    int cost = 0;
    std::vector<double> positive_gains;
    std::optional<Edge> best_edge = std::nullopt;
    double best_gain = 0.0;
};

struct StoredCandidate {
    bool exists = false;
    std::vector<Edge> prefix;
    std::vector<GapTraceEntry> trace;
    Graph graph;
    DenseMatrix katz;
    std::vector<double> scores;
    std::vector<RankedNode> ranking;
    std::vector<std::size_t> top_k;
    std::map<std::string, int> counts;
    double gap = std::numeric_limits<double>::infinity();
    double pair_gap_before = 0.0;
    double pair_gap_after = 0.0;
    std::size_t frontier_order = 0;
};

int dense_step_limit(const ExperimentOptions &options, const Graph &graph) {
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

double pair_gap(const std::vector<double> &scores, std::size_t a, std::size_t b) {
    if (a >= scores.size() || b >= scores.size()) {
        return 0.0;
    }
    return std::max(0.0, scores[b] - scores[a]);
}

double pair_gap_reduction(const StoredCandidate &candidate) {
    return std::max(0.0, candidate.pair_gap_before - candidate.pair_gap_after);
}

bool pair_is_completed(double gap_after) {
    return gap_after <= kEps;
}

GapTraceEntry snapshot_from_state(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts,
    const std::vector<RankedNode> &ranking,
    const std::vector<std::size_t> &top_k,
    const std::map<std::string, int> &counts,
    double gap,
    int step,
    const std::optional<Edge> &edge
) {
    GapTraceEntry entry;
    entry.step = step;
    entry.edge = edge;
    entry.gap = gap;
    entry.katz_mass_objective = katz_mass_objective(graph, scores, target_counts);
    entry.counts = counts;
    entry.top_k = top_k;
    const std::size_t limit = std::min(top_k.size(), ranking.size());
    entry.top_k_ranking.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        entry.top_k_ranking.push_back(ranking[i]);
    }
    return entry;
}

std::vector<FrontierPair> active_frontier_window(
    const Graph &graph,
    const std::vector<RankedNode> &ranking,
    const std::vector<std::size_t> &top_k,
    const std::vector<double> &scores,
    const std::map<std::string, int> &counts,
    const std::map<std::string, int> &target_counts,
    const std::string &promoted_group,
    int frontier_limit
) {
    std::unordered_set<std::size_t> top_k_set(top_k.begin(), top_k.end());
    std::vector<std::size_t> promoted;
    for (const RankedNode &item : ranking) {
        if (top_k_set.find(item.index) != top_k_set.end()) {
            continue;
        }
        if (graph.groups[item.index] == promoted_group) {
            promoted.push_back(item.index);
        }
    }

    std::vector<std::size_t> opposing;
    std::vector<std::size_t> top_k_position(graph.nodes.size(), top_k.size());
    for (std::size_t i = 0; i < top_k.size(); ++i) {
        top_k_position[top_k[i]] = i;
    }
    for (std::size_t node : top_k) {
        if (graph.groups[node] != promoted_group) {
            opposing.push_back(node);
        }
    }
    std::sort(opposing.begin(), opposing.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (std::abs(scores[lhs] - scores[rhs]) > kEps) {
            return scores[lhs] < scores[rhs];
        }
        const auto group_surplus = [&](std::size_t node) {
            const std::string &group = graph.groups[node];
            const int have = counts.count(group) ? counts.at(group) : 0;
            const int want = target_counts.count(group) ? target_counts.at(group) : 0;
            return have - want;
        };
        const int lhs_surplus = group_surplus(lhs);
        const int rhs_surplus = group_surplus(rhs);
        if (lhs_surplus != rhs_surplus) {
            return lhs_surplus > rhs_surplus;
        }
        return top_k_position[lhs] > top_k_position[rhs];
    });

    std::vector<FrontierPair> pairs;
    const bool has_limit = frontier_limit > 0;
    for (std::size_t a : promoted) {
        for (std::size_t b : opposing) {
            pairs.push_back(FrontierPair{a, b, pairs.size()});
            if (has_limit && static_cast<int>(pairs.size()) >= frontier_limit) {
                return pairs;
            }
        }
    }
    return pairs;
}

PairScan scan_pair_gains(
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    double alpha,
    const FrontierPair &pair,
    const std::vector<Edge> &pool
) {
    PairScan result;
    const double gap = pair_gap(scores, pair.a, pair.b);
    result.finite = true;
    if (gap <= kEps) {
        result.cost = 0;
        return result;
    }

    result.positive_gains.reserve(pool.size());
    for (const Edge &edge : pool) {
        const double gain = differential_gain(katz, scores, alpha, pair.a, pair.b, edge.u, edge.v);
        if (gain > kEps) {
            result.positive_gains.push_back(gain);
            if (!result.best_edge.has_value() ||
                gain > result.best_gain + kEps ||
                (std::abs(gain - result.best_gain) <= kEps &&
                 std::make_pair(edge.u, edge.v) < std::make_pair(result.best_edge->u, result.best_edge->v))) {
                result.best_edge = edge;
                result.best_gain = gain;
            }
        }
    }
    std::sort(result.positive_gains.begin(), result.positive_gains.end(), std::greater<double>());

    double cumulative = 0.0;
    for (std::size_t i = 0; i < result.positive_gains.size(); ++i) {
        cumulative += result.positive_gains[i];
        if (cumulative + kEps >= gap) {
            result.cost = static_cast<int>(i + 1);
            return result;
        }
    }

    result.finite = false;
    result.cost = 0;
    return result;
}

std::optional<Edge> best_pair_edge(
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    double alpha,
    const FrontierPair &pair,
    const std::vector<Edge> &pool,
    double &best_gain
) {
    std::optional<Edge> best = std::nullopt;
    best_gain = 0.0;
    for (const Edge &edge : pool) {
        const double gain = differential_gain(katz, scores, alpha, pair.a, pair.b, edge.u, edge.v);
        if (gain > best_gain + kEps ||
            (std::abs(gain - best_gain) <= kEps &&
             best.has_value() &&
             std::make_pair(edge.u, edge.v) < std::make_pair(best->u, best->v))) {
            best = edge;
            best_gain = gain;
        } else if (!best.has_value() && gain > kEps) {
            best = edge;
            best_gain = gain;
        }
    }
    return best;
}

void remove_edge_from_pool(std::vector<Edge> &pool, const Edge &selected) {
    pool.erase(std::remove_if(pool.begin(), pool.end(), [&](const Edge &edge) {
        return edge.u == selected.u && edge.v == selected.v;
    }), pool.end());
}

std::size_t edge_key(const Edge &edge, std::size_t n) {
    return edge.u * n + edge.v;
}

void remove_edges_from_pool(std::vector<Edge> &pool, const std::vector<Edge> &selected, std::size_t n) {
    if (selected.empty()) {
        return;
    }
    std::unordered_set<std::size_t> selected_keys;
    selected_keys.reserve(selected.size());
    for (const Edge &edge : selected) {
        selected_keys.insert(edge_key(edge, n));
    }
    pool.erase(std::remove_if(pool.begin(), pool.end(), [&](const Edge &edge) {
        return selected_keys.find(edge_key(edge, n)) != selected_keys.end();
    }), pool.end());
}

StoredCandidate make_candidate(
    const std::vector<Edge> &prefix,
    const std::vector<GapTraceEntry> &trace,
    const Graph &graph,
    const DenseMatrix &katz,
    const std::vector<double> &scores,
    const std::vector<RankedNode> &ranking,
    const std::vector<std::size_t> &top_k,
    const std::map<std::string, int> &counts,
    double gap,
    double pair_gap_before,
    double pair_gap_after,
    std::size_t frontier_order
) {
    StoredCandidate candidate;
    candidate.exists = true;
    candidate.prefix = prefix;
    candidate.trace = trace;
    candidate.graph = graph;
    candidate.katz = katz;
    candidate.scores = scores;
    candidate.ranking = ranking;
    candidate.top_k = top_k;
    candidate.counts = counts;
    candidate.gap = gap;
    candidate.pair_gap_before = pair_gap_before;
    candidate.pair_gap_after = pair_gap_after;
    candidate.frontier_order = frontier_order;
    return candidate;
}

bool better_objective_candidate(
    const StoredCandidate &candidate,
    const StoredCandidate &incumbent
) {
    if (!candidate.exists) {
        return false;
    }
    if (!incumbent.exists) {
        return true;
    }
    if (candidate.prefix.size() != incumbent.prefix.size()) {
        return candidate.prefix.size() < incumbent.prefix.size();
    }
    if (std::abs(candidate.gap - incumbent.gap) > kEps) {
        return candidate.gap < incumbent.gap;
    }
    const double candidate_reduction = pair_gap_reduction(candidate);
    const double incumbent_reduction = pair_gap_reduction(incumbent);
    if (std::abs(candidate_reduction - incumbent_reduction) > kEps) {
        return candidate_reduction > incumbent_reduction;
    }
    return candidate.frontier_order < incumbent.frontier_order;
}

bool better_plateau_candidate(
    const StoredCandidate &candidate,
    const StoredCandidate &incumbent
) {
    if (!candidate.exists || candidate.prefix.empty() || !pair_is_completed(candidate.pair_gap_after)) {
        return false;
    }
    if (!incumbent.exists || !pair_is_completed(incumbent.pair_gap_after)) {
        return true;
    }
    if (candidate.prefix.size() != incumbent.prefix.size()) {
        return candidate.prefix.size() < incumbent.prefix.size();
    }
    const double candidate_reduction = pair_gap_reduction(candidate);
    const double incumbent_reduction = pair_gap_reduction(incumbent);
    const double candidate_rate = candidate_reduction / static_cast<double>(candidate.prefix.size());
    const double incumbent_rate = incumbent_reduction / static_cast<double>(incumbent.prefix.size());
    if (candidate_rate > incumbent_rate + kEps) {
        return true;
    }
    if (incumbent_rate > candidate_rate + kEps) {
        return false;
    }
    if (candidate_reduction > incumbent_reduction + kEps) {
        return true;
    }
    if (incumbent_reduction > candidate_reduction + kEps) {
        return false;
    }
    return candidate.frontier_order < incumbent.frontier_order;
}

void append_committed_trace(
    std::vector<GapTraceEntry> &target,
    const std::vector<GapTraceEntry> &prefix_trace,
    int step_offset
) {
    for (std::size_t i = 0; i < prefix_trace.size(); ++i) {
        GapTraceEntry entry = prefix_trace[i];
        entry.step = step_offset + static_cast<int>(i) + 1;
        target.push_back(std::move(entry));
    }
}

}  // namespace

AlgorithmResult run_dense_algorithm(
    const Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    Graph working_graph = graph;
    AlgorithmResult result;
    const UpdateModeSelection update_mode = resolve_update_mode(options.update_mode, FairAlgorithmKind::Dense);
    const double target_tolerance = options.epsilon + kEps;
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

    DenseMatrix katz = std::move(initial.centrality.katz);
    std::vector<double> scores = std::move(initial.centrality.scores);
    std::vector<RankedNode> ranking = std::move(initial.ranking);
    std::vector<std::size_t> top_k = std::move(initial.top_k);
    std::map<std::string, int> counts = initial.counts;
    double gap = initial.gap;
    std::string stop_reason = "budget was exhausted.";
    const int step_limit = dense_step_limit(options, working_graph);
    std::vector<Edge> base_pool = candidate_edges(
        working_graph,
        std::nullopt,
        std::nullopt,
        options.edge_admissibility
    );

    while (gap > target_tolerance && static_cast<int>(result.edges.size()) < step_limit) {
        const int budget_remainder = step_limit - static_cast<int>(result.edges.size());
        const auto promoted_group = underrepresented_group(counts, options.target_counts);
        if (!promoted_group.has_value()) {
            stop_reason = "no underrepresented target group was found.";
            break;
        }

        const std::vector<FrontierPair> frontier = active_frontier_window(
            working_graph,
            ranking,
            top_k,
            scores,
            counts,
            options.target_counts,
            *promoted_group,
            options.frontier_limit
        );
        if (frontier.empty()) {
            stop_reason = "no active top-k frontier pair was found.";
            break;
        }

        if (base_pool.empty()) {
            stop_reason = "no admissible edge remains.";
            break;
        }

        StoredCandidate best_improve;
        StoredCandidate best_plateau;
        for (const FrontierPair &pair : frontier) {
            const PairScan initial_scan = scan_pair_gains(
                katz,
                scores,
                initial.centrality.alpha,
                pair,
                base_pool
            );
            if (!initial_scan.finite || initial_scan.cost == 0) {
                continue;
            }

            const int max_steps = std::min(initial_scan.cost, budget_remainder);
            if (max_steps <= 0) {
                continue;
            }

            Graph local_graph = working_graph;
            DenseMatrix local_katz = katz;
            std::vector<double> local_scores = scores;
            std::vector<Edge> local_pool = base_pool;
            std::vector<Edge> prefix;
            std::vector<GapTraceEntry> trace;
            std::vector<RankedNode> local_ranking = ranking;
            std::vector<std::size_t> local_top_k = top_k;
            std::map<std::string, int> local_counts = counts;
            double local_gap = gap;
            const double pair_gap_before = pair_gap(scores, pair.a, pair.b);
            StoredCandidate local_plateau;
            StoredCandidate local_improve;

            for (int step = 0; step < max_steps; ++step) {
                double best_gain = initial_scan.best_gain;
                std::optional<Edge> selected = initial_scan.best_edge;
                if (step > 0) {
                    selected = best_pair_edge(
                        local_katz,
                        local_scores,
                        initial.centrality.alpha,
                        pair,
                        local_pool,
                        best_gain
                    );
                }
                if (!selected.has_value() || best_gain <= kEps) {
                    break;
                }

                if (!apply_update_mode(
                        update_mode,
                        local_katz,
                        local_scores,
                        local_graph,
                        selected->u,
                        selected->v,
                        initial.centrality.alpha
                    )) {
                    break;
                }
                if (!local_graph.add_edge(selected->u, selected->v)) {
                    break;
                }
                remove_edge_from_pool(local_pool, *selected);
                prefix.push_back(*selected);

                local_ranking = tie_aware_full_ranking(
                    local_graph,
                    local_scores,
                    options.k,
                    options.target_counts
                );
                local_top_k = tie_aware_top_k_nodes(
                    local_graph,
                    local_scores,
                    options.k,
                    options.target_counts
                );
                local_counts = group_counts(local_graph, local_top_k);
                local_gap = fairness_gap(local_counts, options.target_counts);
                const double local_pair_gap = pair_gap(local_scores, pair.a, pair.b);
                trace.push_back(snapshot_from_state(
                    local_graph,
                    local_scores,
                    options.target_counts,
                    local_ranking,
                    local_top_k,
                    local_counts,
                    local_gap,
                    static_cast<int>(prefix.size()),
                    *selected
                ));

                if (local_gap < gap - kEps) {
                    local_improve = make_candidate(
                        prefix,
                        trace,
                        local_graph,
                        local_katz,
                        local_scores,
                        local_ranking,
                        local_top_k,
                        local_counts,
                        local_gap,
                        pair_gap_before,
                        local_pair_gap,
                        pair.order
                    );
                    break;
                }

                if (local_gap <= gap + kEps && local_pair_gap < pair_gap_before - kEps) {
                    StoredCandidate candidate = make_candidate(
                        prefix,
                        trace,
                        local_graph,
                        local_katz,
                        local_scores,
                        local_ranking,
                        local_top_k,
                        local_counts,
                        local_gap,
                        pair_gap_before,
                        local_pair_gap,
                        pair.order
                    );
                    if (better_plateau_candidate(candidate, local_plateau)) {
                        local_plateau = std::move(candidate);
                    }
                }
            }

            if (better_objective_candidate(local_improve, best_improve)) {
                best_improve = std::move(local_improve);
            }
            if (better_plateau_candidate(local_plateau, best_plateau)) {
                best_plateau = std::move(local_plateau);
            }
        }

        const StoredCandidate *chosen = nullptr;
        if (best_improve.exists) {
            chosen = &best_improve;
        } else if (best_plateau.exists) {
            chosen = &best_plateau;
        }
        if (chosen == nullptr) {
            stop_reason = "no committable frontier prefix was found.";
            break;
        }

        const int step_offset = static_cast<int>(result.edges.size());
        result.edges.insert(result.edges.end(), chosen->prefix.begin(), chosen->prefix.end());
        append_committed_trace(result.gap_trace, chosen->trace, step_offset);
        remove_edges_from_pool(base_pool, chosen->prefix, working_graph.nodes.size());
        working_graph = chosen->graph;
        katz = chosen->katz;
        scores = chosen->scores;
        ranking = chosen->ranking;
        top_k = chosen->top_k;
        counts = chosen->counts;
        gap = chosen->gap;
        if (gap <= target_tolerance) {
            result.found = true;
            result.final_scores = scores;
            return result;
        }
    }

    if (gap > target_tolerance && static_cast<int>(result.edges.size()) >= step_limit) {
        stop_reason = "budget was exhausted.";
    }
    result.message = "dense stopped before reaching the target within the budget: " + stop_reason;
    result.final_scores = scores;
    return result;
}

}  // namespace top_k

#include "algorithms/helpers/helpers.h"
#include "algorithms/helpers/updates.h"
#include "src/algorithms.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace top_k {
namespace {

int mass_budget_remainder(const ExperimentOptions &options, const Graph &graph) {
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

int katz_mass_batch_attempts(const ExperimentOptions &options, std::size_t initial_edge_count) {
    if (options.katz_mass_attempt_fraction <= 0.0) {
        return options.katz_mass_attempts_per_commit;
    }
    const double raw_attempts =
        std::ceil(static_cast<double>(initial_edge_count) *
                  options.katz_mass_attempt_fraction);
    if (raw_attempts <= 1.0) {
        return 1;
    }
    const double capped = std::min<double>(
        raw_attempts,
        static_cast<double>(std::numeric_limits<int>::max())
    );
    return static_cast<int>(capped);
}

double protected_mass_total(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
) {
    const auto protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value()) {
        return 0.0;
    }
    double total = 0.0;
    for (std::size_t node = 0; node < scores.size(); ++node) {
        if (graph.groups[node] == *protected_group) {
            total += scores[node];
        }
    }
    return total;
}

std::vector<std::size_t> directional_mass_targets(
    const Graph &graph,
    const std::string &protected_group,
    bool protected_needs_more_mass,
    const std::string &edge_admissibility
) {
    std::vector<std::size_t> targets;
    targets.reserve(graph.nodes.size());
    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        const bool node_is_protected = graph.groups[node] == protected_group;
        if (protected_needs_more_mass != node_is_protected) {
            continue;
        }
        if (has_addable_source_to_target(graph, node, edge_admissibility)) {
            targets.push_back(node);
        }
    }
    return targets;
}

std::optional<Edge> choose_directional_mass_edge(
    const Graph &graph,
    const std::vector<std::size_t> &targets,
    const std::string &edge_admissibility,
    std::mt19937 &rng
) {
    if (graph.nodes.size() < 2 || targets.empty()) {
        return std::nullopt;
    }

    std::uniform_int_distribution<std::size_t> target_dist(0, targets.size() - 1);
    std::uniform_int_distribution<std::size_t> node_dist(0, graph.nodes.size() - 1);
    const std::size_t target = targets[target_dist(rng)];
    const std::vector<char> addable_sources =
        addable_sources_to_target(graph, target, edge_admissibility);

    const std::size_t start = node_dist(rng);
    for (std::size_t offset = 0; offset < graph.nodes.size(); ++offset) {
        const std::size_t source = (start + offset) % graph.nodes.size();
        if (source < addable_sources.size() && addable_sources[source] != 0) {
            return Edge{source, target};
        }
    }
    return std::nullopt;
}

}  // namespace

AlgorithmResult run_katz_mass_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    AlgorithmResult result;
    const UpdateModeSelection update_mode =
        resolve_update_mode(options.update_mode, FairAlgorithmKind::KatzMass);
    const double target_tolerance = options.epsilon * options.epsilon + kEps;
    const double alpha = initial.centrality.alpha;
    std::mt19937 rng(options.seed);

    result.algorithm_resolved = "katz_mass_random";
    result.update_mode_resolved = update_mode.resolved;
    result.gap_trace.push_back(snapshot(
        graph,
        initial.centrality.scores,
        options.k,
        options.target_counts,
        0,
        std::nullopt
    ));
    const double gap_tolerance = options.epsilon + kEps;
    if (result.gap_trace.back().gap <= gap_tolerance) {
        result.found = true;
        result.already_satisfied = true;
        result.final_scores = initial.centrality.scores;
        return result;
    }

    DenseMatrix katz = std::move(initial.centrality.katz);
    std::vector<double> scores = std::move(initial.centrality.scores);
    double current_objective = katz_mass_objective(graph, scores, options.target_counts);

    const std::size_t initial_edge_count = graph.edge_count();
    const int batch_attempts = katz_mass_batch_attempts(options, initial_edge_count);
    int budget_remainder = mass_budget_remainder(options, graph);
    int batches_completed = 0;
    std::string stop_reason = "loop condition ended.";
    while (budget_remainder > 0) {
        if (options.katz_mass_max_commits.has_value() &&
            batches_completed >= *options.katz_mass_max_commits) {
            stop_reason = "Katz mass commit limit was reached.";
            break;
        }
        if (current_objective <= target_tolerance) {
            stop_reason = "Katz mass target tolerance was reached.";
            break;
        }
        std::vector<Edge> batch;
        batch.reserve(static_cast<std::size_t>(
            std::min(batch_attempts, budget_remainder)
        ));
        const double target_share = protected_group_target_share(options.target_counts);
        const auto protected_group = protected_group_name(options.target_counts);
        const double total_mass = std::accumulate(scores.begin(), scores.end(), 0.0);
        const double protected_mass = protected_mass_total(
            graph,
            scores,
            options.target_counts
        );
        if (!protected_group.has_value() || std::abs(total_mass) <= kEps) {
            stop_reason = "Katz mass direction could not be resolved.";
            break;
        }
        const double current_share = protected_mass / total_mass;
        const bool protected_needs_more_mass = current_share < target_share - kEps;
        const bool protected_needs_less_mass = current_share > target_share + kEps;
        if (!protected_needs_more_mass && !protected_needs_less_mass) {
            stop_reason = "Katz mass target tolerance was reached.";
            break;
        }
        const std::vector<std::size_t> targets = directional_mass_targets(
            graph,
            *protected_group,
            protected_needs_more_mass,
            options.edge_admissibility
        );
        if (targets.empty()) {
            stop_reason = "no admissible target had an addable source in the target direction.";
            break;
        }
        const int attempt_limit = std::min(batch_attempts, budget_remainder);
        for (int attempt = 0; attempt < attempt_limit; ++attempt) {
            ++result.candidate_attempts;
            std::optional<Edge> selected = choose_directional_mass_edge(
                graph,
                targets,
                options.edge_admissibility,
                rng
            );
            if (!selected.has_value()) {
                continue;
            }
            if (graph.add_edge(selected->u, selected->v)) {
                batch.push_back(*selected);
                result.edges.push_back(*selected);
                --budget_remainder;
            }
        }
        if (batch.empty()) {
            if (stop_reason == "loop condition ended.") {
                stop_reason = "no sampled edge moved Katz mass in the target direction.";
            }
            break;
        }

        if (update_mode.kind == UpdateModeKind::JacobiScores) {
            scores = jacobi_scores(graph, alpha, scores);
        } else {
            bool update_failed = false;
            for (const Edge &edge : batch) {
                if (!apply_update_mode(
                        update_mode,
                        katz,
                        scores,
                        graph,
                        edge.u,
                        edge.v,
                        alpha
                    )) {
                    update_failed = true;
                    break;
                }
            }
            if (update_failed) {
                stop_reason = "failed to apply Katz mass update.";
                break;
            }
        }

        current_objective = katz_mass_objective(graph, scores, options.target_counts);
        ++batches_completed;
        result.gap_trace.push_back(snapshot(
            graph,
            scores,
            options.k,
            options.target_counts,
            static_cast<int>(result.edges.size()),
            batch.back()
        ));
        if (result.gap_trace.back().gap <= gap_tolerance) {
            result.found = true;
            stop_reason = "top-k target tolerance was reached.";
            break;
        }
        if (current_objective <= target_tolerance) {
            stop_reason = "Katz mass target tolerance was reached.";
            break;
        }
    }

    if (result.edges.empty() && result.found) {
        result.already_satisfied = true;
    }
    if (budget_remainder <= 0) {
        stop_reason = "budget was exhausted.";
    }
    result.final_scores = scores;
    std::ostringstream message;
    message << "KatzMass-Random sampled direct-target edges in the current Katz-mass "
            << "correction direction using "
            << options.edge_admissibility
            << " admissibility, did not scan the full missing-edge set, and stopped because "
            << stop_reason
            << " final_mass_objective=" << current_objective << "."
            << " attempts_per_commit=" << batch_attempts << "."
            << " commits_completed=" << batches_completed << "."
            << " candidate_attempts=" << result.candidate_attempts << ".";
    if (options.katz_mass_attempt_fraction > 0.0) {
        message << " attempt_fraction=" << options.katz_mass_attempt_fraction << ".";
    }
    result.message = message.str();
    return result;
}

}  // namespace top_k

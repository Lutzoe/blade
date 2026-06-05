#include "src/algorithms.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace top_k {

AlgorithmSelection resolve_algorithm_selection(const std::string &raw) {
    AlgorithmSelection selection;
    if (raw == "gap_greedy") {
        selection.kind = FairAlgorithmKind::GapGreedy;
        selection.resolved = "gap_greedy";
        return selection;
    }
    if (raw == "katz_mass") {
        selection.kind = FairAlgorithmKind::KatzMass;
        selection.resolved = "katz_mass";
        return selection;
    }
    if (raw == "optimal") {
        selection.kind = FairAlgorithmKind::Optimal;
        selection.resolved = "optimal";
        return selection;
    }
    if (raw == "dense") {
        selection.kind = FairAlgorithmKind::Dense;
        selection.resolved = "dense";
        return selection;
    }
    if (raw == "blade") {
        selection.kind = FairAlgorithmKind::Blade;
        selection.resolved = "blade";
        return selection;
    }
    if (raw == "blade_no_batch") {
        selection.kind = FairAlgorithmKind::BladeNoBatch;
        selection.resolved = "blade_no_batch";
        return selection;
    }
    if (raw == "same_group_support" || raw == "same_group") {
        selection.kind = FairAlgorithmKind::SameGroupSupport;
        selection.resolved = "same_group_support";
        return selection;
    }
    throw std::runtime_error(
        "algorithm must be one of: gap_greedy, katz_mass, optimal, dense, "
        "blade, blade_no_batch, same_group_support"
    );
}

AlgorithmResult run_fair_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
) {
    const AlgorithmSelection selection = resolve_algorithm_selection(options.algorithm);
    const UpdateModeSelection update_mode = resolve_update_mode(options.update_mode, selection.kind);
    const auto started = std::chrono::steady_clock::now();

    AlgorithmResult result;
    result.algorithm_resolved = selection.resolved;
    result.update_mode_resolved = update_mode.resolved;
    result.message = selection.message;
    if (!update_mode.message.empty()) {
        if (!result.message.empty()) {
            result.message += " ";
        }
        result.message += update_mode.message;
    }

    if (selection.kind != FairAlgorithmKind::Blade &&
        selection.kind != FairAlgorithmKind::BladeNoBatch &&
        selection.kind != FairAlgorithmKind::KatzMass &&
        selection.kind != FairAlgorithmKind::SameGroupSupport &&
        update_mode.kind != UpdateModeKind::JacobiScores &&
        initial.centrality.katz.n == 0 && options.budget > 0 && initial.gap > kEps) {
        throw std::runtime_error(
            "The selected backend provides Katz centrality scores only. "
            "This runner's fair-update algorithms still require the full Katz matrix for "
            "differential gain and update steps. Use eigen_direct or a matrix-based "
            "update mode."
        );
    }

    AlgorithmResult computed;
    if (selection.kind == FairAlgorithmKind::GapGreedy) {
        computed = run_gap_greedy_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::KatzMass) {
        computed = run_katz_mass_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::Optimal) {
        computed = run_optimal_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::Dense) {
        computed = run_dense_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::Blade) {
        computed = run_blade_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::BladeNoBatch) {
        computed = run_blade_no_batch_algorithm(graph, std::move(initial), options);
    } else if (selection.kind == FairAlgorithmKind::SameGroupSupport) {
        computed = run_same_group_support_algorithm(graph, std::move(initial), options);
    } else {
        throw std::runtime_error("unsupported algorithm selection");
    }

    result.found = computed.found;
    result.already_satisfied = computed.already_satisfied;
    result.edges = std::move(computed.edges);
    result.gap_trace = std::move(computed.gap_trace);
    result.final_scores = std::move(computed.final_scores);
    result.candidate_attempts = computed.candidate_attempts;
    result.algorithm_resolved = computed.algorithm_resolved.empty()
        ? result.algorithm_resolved
        : computed.algorithm_resolved;
    result.update_mode_resolved = computed.update_mode_resolved.empty()
        ? result.update_mode_resolved
        : computed.update_mode_resolved;
    if (!computed.message.empty()) {
        if (!result.message.empty()) {
            result.message += " ";
        }
        result.message += computed.message;
    }

    const auto finished = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(finished - started).count();
    return result;
}

}  // namespace top_k

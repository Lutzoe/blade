#include "katz_calculation/katz.h"
#include "algorithms/helpers/updates.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace top_k {
namespace {

std::string canonicalize_token(const std::string &raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return normalized;
}

std::vector<double> column_sums(const DenseMatrix &matrix) {
    std::vector<double> sums(matrix.n, 0.0);
    for (std::size_t row = 0; row < matrix.n; ++row) {
        for (std::size_t col = 0; col < matrix.n; ++col) {
            sums[col] += matrix.at(row, col);
        }
    }
    return sums;
}

bool uses_score_only_jacobi(const ExperimentOptions &options) {
    const AlgorithmSelection algorithm_selection = resolve_algorithm_selection(options.algorithm);
    const UpdateModeSelection update_mode =
        resolve_update_mode(options.update_mode, algorithm_selection.kind);
    const FairAlgorithmKind kind = algorithm_selection.kind;
    const bool jacobi_requested = update_mode.kind == UpdateModeKind::JacobiScores;
    if (jacobi_requested) {
        return kind == FairAlgorithmKind::Blade ||
            kind == FairAlgorithmKind::BladeNoBatch ||
            kind == FairAlgorithmKind::GapGreedy ||
            kind == FairAlgorithmKind::KatzMass ||
            kind == FairAlgorithmKind::Optimal ||
            kind == FairAlgorithmKind::SameGroupSupport;
    }
    return false;
}

}  // namespace

BackendSelection resolve_backend_selection(const std::string &raw) {
    BackendSelection selection;
    const std::string key = canonicalize_token(raw);
    if (key == "eigendirect") {
        selection.kind = KatzBackendKind::EigenDirect;
        selection.resolved = "eigen_direct";
        return selection;
    }
    throw std::runtime_error(
        "backend must be: eigen_direct"
    );
}

CentralityComputation compute_katz_centrality(
    const Graph &graph,
    const ExperimentOptions &options,
    std::optional<double> alpha_override
) {
    CentralityComputation computation;
    computation.alpha = alpha_override.value_or(resolve_alpha(graph, options.dataset));
    computation.backend = resolve_backend_selection(options.backend);

    const auto started = std::chrono::steady_clock::now();
    if (uses_score_only_jacobi(options)) {
        computation.scores = jacobi_scores(graph, computation.alpha, {});
        computation.backend.resolved += "(score_only_jacobi)";
    } else if (computation.backend.kind == KatzBackendKind::EigenDirect) {
        computation.katz = compute_eigen_direct_katz_matrix(graph, computation.alpha);
    }
    const auto finished = std::chrono::steady_clock::now();

    computation.elapsed_seconds = std::chrono::duration<double>(finished - started).count();
    if (computation.scores.empty()) {
        computation.scores = column_sums(computation.katz);
    }
    return computation;
}

}  // namespace top_k

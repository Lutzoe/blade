#include "algorithms/helpers/updates.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <utility>

namespace top_k {
namespace {

inline constexpr int kJacobiMaxIterations = 200;
inline constexpr double kJacobiTolerance = 1e-10;
inline constexpr double kIncrementalKatzResidualTolerance = 1e-11;
inline constexpr std::size_t kIncrementalKatzMaxRelaxations = 50000000;

UpdateModeSelection default_update_mode_for_algorithm(FairAlgorithmKind algorithm_kind) {
    UpdateModeSelection selection;
    if (algorithm_kind == FairAlgorithmKind::GapGreedy ||
        algorithm_kind == FairAlgorithmKind::Blade ||
        algorithm_kind == FairAlgorithmKind::BladeNoBatch ||
        algorithm_kind == FairAlgorithmKind::KatzMass ||
        algorithm_kind == FairAlgorithmKind::Optimal ||
        algorithm_kind == FairAlgorithmKind::SameGroupSupport) {
        selection.kind = UpdateModeKind::JacobiScores;
        selection.resolved = "jacobi_scores(default)";
        return selection;
    }
    selection.kind = UpdateModeKind::ShermanMorrison;
    selection.resolved = "sherman_morrison(default)";
    return selection;
}

}  // namespace

UpdateModeSelection resolve_update_mode(
    const std::string &raw,
    FairAlgorithmKind algorithm_kind
) {
    if (raw.empty()) {
        return default_update_mode_for_algorithm(algorithm_kind);
    }

    UpdateModeSelection selection;
    if (algorithm_kind == FairAlgorithmKind::Optimal && raw != "jacobi") {
        throw std::runtime_error("optimal uses jacobi update mode only");
    }
    if (algorithm_kind == FairAlgorithmKind::SameGroupSupport && raw != "jacobi") {
        throw std::runtime_error("same_group_support uses jacobi update mode only");
    }
    if (raw == "sherman_morrison" &&
        algorithm_kind != FairAlgorithmKind::Blade &&
        algorithm_kind != FairAlgorithmKind::BladeNoBatch &&
        algorithm_kind != FairAlgorithmKind::Optimal) {
        selection.kind = UpdateModeKind::ShermanMorrison;
        selection.resolved = "sherman_morrison";
        return selection;
    }
    if ((algorithm_kind == FairAlgorithmKind::GapGreedy ||
         algorithm_kind == FairAlgorithmKind::Optimal ||
         algorithm_kind == FairAlgorithmKind::KatzMass ||
         algorithm_kind == FairAlgorithmKind::Blade ||
         algorithm_kind == FairAlgorithmKind::BladeNoBatch ||
         algorithm_kind == FairAlgorithmKind::SameGroupSupport) &&
        raw == "jacobi") {
        selection.kind = UpdateModeKind::JacobiScores;
        selection.resolved = "jacobi_scores";
        return selection;
    }
    throw std::runtime_error(
        algorithm_kind == FairAlgorithmKind::Optimal
            ? "update mode must be: jacobi"
        : (algorithm_kind == FairAlgorithmKind::SameGroupSupport)
            ? "update mode must be: jacobi"
        : (algorithm_kind == FairAlgorithmKind::KatzMass)
            ? "update mode must be one of: jacobi, sherman_morrison"
            : (algorithm_kind == FairAlgorithmKind::GapGreedy)
                ? "update mode must be one of: jacobi, sherman_morrison"
            : (algorithm_kind == FairAlgorithmKind::Blade ||
               algorithm_kind == FairAlgorithmKind::BladeNoBatch)
                ? "update mode must be: jacobi"
            : "update mode must be: sherman_morrison"
    );
}

bool sherman_morrison_update(
    DenseMatrix &katz,
    std::vector<double> &scores,
    std::size_t u,
    std::size_t v,
    double alpha
) {
    double eta = 1.0 - alpha * katz.at(v, u);
    if (eta <= kEps) {
        return false;
    }
    if (1.0 - eta <= kEps) {
        eta = 1.0;
    }
    std::vector<double> col_u(katz.n, 0.0);
    std::vector<double> row_v(katz.n, 0.0);
    for (std::size_t i = 0; i < katz.n; ++i) {
        col_u[i] = katz.at(i, u);
        row_v[i] = katz.at(v, i);
    }
    const double factor = alpha / eta;
    for (std::size_t i = 0; i < katz.n; ++i) {
        for (std::size_t j = 0; j < katz.n; ++j) {
            katz.at(i, j) += factor * col_u[i] * row_v[j];
        }
    }
    for (std::size_t j = 0; j < scores.size(); ++j) {
        scores[j] += factor * scores[u] * row_v[j];
    }
    return true;
}

std::vector<double> jacobi_scores(
    const Graph &graph,
    double alpha,
    const std::vector<double> &warm_start
) {
    const std::size_t n = graph.nodes.size();
    std::vector<double> current = warm_start.size() == n
        ? warm_start
        : std::vector<double>(n, 1.0);
    std::vector<double> next(n, 1.0);

    for (int iter = 0; iter < kJacobiMaxIterations; ++iter) {
        double max_delta = 0.0;
        for (std::size_t v = 0; v < n; ++v) {
            double value = 1.0;
            for (std::size_t u : graph.in_neighbors[v]) {
                value += alpha * current[u];
            }
            next[v] = value;
            max_delta = std::max(max_delta, std::abs(value - current[v]));
        }
        current.swap(next);
        if (max_delta <= kJacobiTolerance) {
            break;
        }
    }
    return current;
}

bool incremental_katz_scores_after_edges(
    const Graph &graph,
    double alpha,
    const std::vector<Edge> &added_edges,
    std::vector<double> &scores
) {
    const std::size_t n = graph.nodes.size();
    if (scores.size() != n || alpha <= 0.0 || added_edges.empty()) {
        return scores.size() == n;
    }

    std::vector<double> residual(n, 0.0);
    std::vector<char> queued(n, 0);
    std::deque<std::size_t> queue;
    auto enqueue = [&](std::size_t node) {
        if (node >= n || queued[node] != 0 ||
            std::abs(residual[node]) <= kIncrementalKatzResidualTolerance) {
            return;
        }
        queued[node] = 1;
        queue.push_back(node);
    };

    for (const Edge &edge : added_edges) {
        if (edge.u >= n || edge.v >= n) {
            return false;
        }
        residual[edge.v] += alpha * scores[edge.u];
        enqueue(edge.v);
    }

    std::size_t relaxations = 0;
    while (!queue.empty()) {
        if (++relaxations > kIncrementalKatzMaxRelaxations) {
            return false;
        }
        const std::size_t node = queue.front();
        queue.pop_front();
        queued[node] = 0;
        const double delta = residual[node];
        residual[node] = 0.0;
        if (std::abs(delta) <= kIncrementalKatzResidualTolerance) {
            continue;
        }
        scores[node] += delta;
        const double propagated = alpha * delta;
        if (std::abs(propagated) <= kIncrementalKatzResidualTolerance) {
            continue;
        }
        for (std::size_t next : graph.out_neighbors[node]) {
            residual[next] += propagated;
            enqueue(next);
        }
    }
    return true;
}

bool apply_update_mode(
    const UpdateModeSelection &selection,
    DenseMatrix &katz,
    std::vector<double> &scores,
    const Graph &graph,
    std::size_t u,
    std::size_t v,
    double alpha
) {
    if (selection.kind == UpdateModeKind::ShermanMorrison) {
        return sherman_morrison_update(katz, scores, u, v, alpha);
    }
    if (selection.kind == UpdateModeKind::JacobiScores) {
        Graph updated_graph = graph;
        if (!updated_graph.add_edge(u, v)) {
            return false;
        }
        scores = jacobi_scores(updated_graph, alpha, scores);
        return true;
    }
    return false;
}

}  // namespace top_k

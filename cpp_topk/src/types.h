#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace top_k {

inline constexpr double kEps = 1e-12;
inline constexpr int kDefaultBladeFrontierLimit = 2;

enum class KatzBackendKind {
    EigenDirect,
};

enum class FairAlgorithmKind {
    GapGreedy,
    KatzMass,
    Optimal,
    Dense,
    Blade,
    BladeNoBatch,
    SameGroupSupport,
};

enum class UpdateModeKind {
    ShermanMorrison,
    JacobiScores,
};

struct DenseMatrix {
    std::size_t n = 0;
    std::vector<double> data;

    DenseMatrix() = default;
    explicit DenseMatrix(std::size_t size) : n(size), data(size * size, 0.0) {}

    double &at(std::size_t row, std::size_t col) {
        return data[row * n + col];
    }

    double at(std::size_t row, std::size_t col) const {
        return data[row * n + col];
    }
};

struct Edge {
    std::size_t u = 0;
    std::size_t v = 0;
};

struct RankedNode {
    std::size_t index = 0;
    double score = 0.0;
};

struct GapTraceEntry {
    int step = 0;
    std::optional<Edge> edge;
    double gap = 0.0;
    double katz_mass_objective = 0.0;
    std::map<std::string, int> counts;
    std::vector<std::size_t> top_k;
    std::vector<RankedNode> top_k_ranking;
};

struct DatasetSpec {
    std::string name;
    std::string edge_path;
    std::string group_path;
    bool directed = false;
};

struct Graph {
    bool directed = false;
    std::vector<std::string> nodes;
    std::vector<std::string> groups;
    std::unordered_map<std::string, std::size_t> index_of;
    std::vector<std::vector<std::size_t>> out_neighbors;
    std::vector<std::vector<std::size_t>> in_neighbors;

    std::size_t edge_count() const {
        std::size_t total = 0;
        for (const auto &neighbors : out_neighbors) {
            total += neighbors.size();
        }
        return total;
    }

    bool has_edge(std::size_t u, std::size_t v) const {
        const auto &neighbors = out_neighbors[u];
        const auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        return it != neighbors.end() && *it == v;
    }

    bool add_edge(std::size_t u, std::size_t v) {
        if (u == v) {
            return false;
        }
        auto &out = out_neighbors[u];
        const auto out_it = std::lower_bound(out.begin(), out.end(), v);
        if (out_it != out.end() && *out_it == v) {
            return false;
        }
        out.insert(out_it, v);

        auto &in = in_neighbors[v];
        const auto in_it = std::lower_bound(in.begin(), in.end(), u);
        in.insert(in_it, u);
        return true;
    }

    int max_out_degree() const {
        int best = 0;
        for (const auto &neighbors : out_neighbors) {
            best = std::max(best, static_cast<int>(neighbors.size()));
        }
        return best;
    }

    int max_in_degree() const {
        int best = 0;
        for (const auto &neighbors : in_neighbors) {
            best = std::max(best, static_cast<int>(neighbors.size()));
        }
        return best;
    }
};

struct BackendSelection {
    KatzBackendKind kind = KatzBackendKind::EigenDirect;
    std::string resolved = "eigen_direct";
    std::string message;
};

struct AlgorithmSelection {
    FairAlgorithmKind kind = FairAlgorithmKind::Blade;
    std::string resolved = "blade";
    std::string message;
};

struct UpdateModeSelection {
    UpdateModeKind kind = UpdateModeKind::ShermanMorrison;
    std::string resolved = "sherman_morrison";
    std::string message;
};

struct CentralityComputation {
    double alpha = 0.1;
    DenseMatrix katz;
    std::vector<double> scores;
    double elapsed_seconds = 0.0;
    BackendSelection backend;
};

struct RankingState {
    std::vector<std::size_t> top_k;
    std::vector<RankedNode> ranking;
    std::vector<RankedNode> top_k_ranking;
    std::map<std::string, int> counts;
    double gap = 0.0;
    CentralityComputation centrality;
};

struct AlgorithmResult {
    bool found = false;
    bool already_satisfied = false;
    std::vector<Edge> edges;
    std::vector<GapTraceEntry> gap_trace;
    std::vector<double> final_scores;
    std::size_t candidate_attempts = 0;
    std::string message;
    std::string algorithm_resolved;
    std::string update_mode_resolved;
    double elapsed_seconds = 0.0;
};

struct ExperimentOptions {
    std::string dataset;
    std::string data_root = "../data";
    std::optional<int> bpa_size;
    std::optional<std::string> bpa_rho;
    std::optional<std::string> edge_path;
    std::optional<std::string> group_path;
    std::optional<bool> directed;
    int k = -1;
    int budget = -1;
    std::map<std::string, int> target_counts;
    std::string backend = "eigen_direct";
    std::string algorithm = "blade";
    std::string update_mode;
    int frontier_limit = 0;
    int gap_greedy_frontier_limit = 0;
    int same_group_window = 100;
    int katz_mass_attempts_per_commit = 1;
    double katz_mass_attempt_fraction = 0.0;
    std::optional<int> katz_mass_max_commits;
    std::string edge_admissibility = "any";
    unsigned int seed = 7;
    std::optional<double> alpha;
    double epsilon = 0.0;
};

struct ExperimentReport {
    DatasetSpec dataset;
    ExperimentOptions options;
    std::size_t initial_edge_count = 0;
    RankingState initial;
    AlgorithmResult fair_result;
    RankingState final;
    double total_elapsed_seconds = 0.0;
};

struct ExperimentSetup {
    ExperimentOptions options;
    DatasetSpec dataset;
    Graph graph;
};

BackendSelection resolve_backend_selection(const std::string &raw);
AlgorithmSelection resolve_algorithm_selection(const std::string &raw);
UpdateModeSelection resolve_update_mode(
    const std::string &raw,
    FairAlgorithmKind algorithm_kind
);

DatasetSpec resolve_dataset(const ExperimentOptions &options);
Graph load_graph(const DatasetSpec &spec);

double resolve_alpha(const Graph &graph, const std::string &dataset_name);
CentralityComputation compute_katz_centrality(
    const Graph &graph,
    const ExperimentOptions &options,
    std::optional<double> alpha_override = std::nullopt
);
std::vector<RankedNode> rank_scores(const std::vector<double> &scores);
std::vector<RankedNode> rank_top_scores(const std::vector<double> &scores, int limit);
std::vector<RankedNode> tie_aware_top_k_ranking(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
);
std::vector<RankedNode> tie_aware_full_ranking(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
);
std::vector<std::size_t> top_k_nodes(const std::vector<RankedNode> &ranking, int k);
std::vector<std::size_t> tie_aware_top_k_nodes(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
);
std::map<std::string, int> group_counts(const Graph &graph, const std::vector<std::size_t> &nodes);
std::optional<std::string> protected_group_name(const std::map<std::string, int> &target_counts);
double protected_group_target_share(const std::map<std::string, int> &target_counts);
double fairness_gap(
    const std::map<std::string, int> &counts,
    const std::map<std::string, int> &target_counts
);
double katz_mass_share(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
);
double katz_mass_objective(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
);
RankingState build_ranking_state(
    const Graph &graph,
    const CentralityComputation &centrality,
    int k,
    const std::map<std::string, int> &target_counts
);
RankingState compute_ranking_state(
    const Graph &graph,
    const ExperimentOptions &options,
    int k,
    const std::map<std::string, int> &target_counts,
    std::optional<double> alpha_override = std::nullopt
);

AlgorithmResult run_fair_algorithm(
    Graph &graph,
    RankingState initial,
    const ExperimentOptions &options
);
ExperimentSetup prepare_experiment_setup(int argc, char **argv);
ExperimentReport run_single_experiment(
    Graph &graph,
    const DatasetSpec &dataset,
    const ExperimentOptions &options
);
int run_cli(int argc, char **argv);

void print_usage();
ExperimentOptions parse_args(int argc, char **argv);
void print_summary(const Graph &graph, const ExperimentReport &report);

}  // namespace top_k

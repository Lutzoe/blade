#include "src/types.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace top_k {
namespace {

inline constexpr int kDefaultGapGreedyFrontierLimit = 2;
inline constexpr std::size_t kMaxPrintedAddedEdges = 10;
inline constexpr std::size_t kMaxPrintedGapTraceRows = 20;
inline constexpr const char *kPrintAllAddedEdgesEnv = "TOPK_PRINT_ALL_ADDED_EDGES";

struct MaxScoreChange {
    std::size_t index = 0;
    double value = -1.0;
};

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

std::vector<std::string> split_csv(const std::string &raw) {
    std::vector<std::string> parts;
    std::stringstream stream(raw);
    std::string piece;
    while (std::getline(stream, piece, ',')) {
        if (!piece.empty()) {
            parts.push_back(piece);
        }
    }
    return parts;
}

std::map<std::string, int> parse_target_counts(const std::string &raw) {
    std::map<std::string, int> result;
    for (const auto &piece : split_csv(raw)) {
        const auto pos = piece.find(':');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid target entry: " + piece);
        }
        const std::string group = piece.substr(0, pos);
        const int count = std::stoi(piece.substr(pos + 1));
        result[group] = count;
    }
    if (result.empty()) {
        throw std::runtime_error("Target counts cannot be empty.");
    }
    return result;
}

int sum_target_counts(const std::map<std::string, int> &counts) {
    int total = 0;
    for (const auto &[_, value] : counts) {
        total += value;
    }
    return total;
}

void print_counts(const std::map<std::string, int> &counts) {
    bool first = true;
    std::cout << "{";
    for (const auto &[group, count] : counts) {
        if (!first) {
            std::cout << ", ";
        }
        first = false;
        std::cout << group << ": " << count;
    }
    std::cout << "}";
}

void print_top_k_rows(
    const Graph &graph,
    const std::vector<RankedNode> &ranking
) {
    for (const auto &item : ranking) {
        std::cout << "  " << graph.nodes[item.index]
                  << " [" << graph.groups[item.index] << "]"
                  << " (" << std::fixed << std::setprecision(6) << item.score << ")\n";
    }
}

void print_added_edge_summary(
    const Graph &graph,
    const std::vector<Edge> &edges
) {
    std::cout << "edges added (" << edges.size() << "):";
    if (edges.empty()) {
        std::cout << "\n";
        return;
    }
    const char *print_all_env = std::getenv(kPrintAllAddedEdgesEnv);
    const bool print_all =
        print_all_env != nullptr && std::string(print_all_env) == "1";
    if (print_all) {
        for (const auto &edge : edges) {
            std::cout << " (" << graph.nodes[edge.u] << ", " << graph.nodes[edge.v] << ")";
        }
        std::cout << "\n";
        return;
    }
    const std::size_t printed_edges =
        std::min(kMaxPrintedAddedEdges, edges.size());
    std::cout << " sample:";
    for (std::size_t i = 0; i < printed_edges; ++i) {
        const auto &edge = edges[i];
        std::cout << " (" << graph.nodes[edge.u] << ", " << graph.nodes[edge.v] << ")";
    }
    if (edges.size() > printed_edges) {
        std::cout << " ...";
    }
    std::cout << "\n";
}

void print_gap_trace_row(
    const Graph &graph,
    const GapTraceEntry &entry,
    int previous_step
) {
    std::cout << "  step " << entry.step << ": gap="
              << std::fixed << std::setprecision(8) << entry.gap
              << ", katz_mass_objective="
              << std::scientific << std::setprecision(12)
              << entry.katz_mass_objective
              << std::fixed;
    if (entry.edge.has_value()) {
        if (entry.step > previous_step + 1) {
            std::cout << ", edge_span=" << (previous_step + 1)
                      << "-" << entry.step
                      << ", last_edge=("
                      << graph.nodes[entry.edge->u] << ", "
                      << graph.nodes[entry.edge->v] << ")";
        } else {
            std::cout << ", edge=("
                      << graph.nodes[entry.edge->u] << ", "
                      << graph.nodes[entry.edge->v] << ")";
        }
    }
    std::cout << ", counts=";
    print_counts(entry.counts);
    std::cout << "\n";
}

void print_gap_trace_summary(
    const Graph &graph,
    const std::vector<GapTraceEntry> &trace
) {
    if (trace.empty()) {
        return;
    }
    std::cout << "unfairness gap snapshot trace (" << trace.size() << " entries):\n";
    if (trace.size() <= kMaxPrintedGapTraceRows) {
        int previous_step = 0;
        for (const auto &entry : trace) {
            print_gap_trace_row(graph, entry, previous_step);
            previous_step = entry.step;
        }
        return;
    }

    const std::size_t prefix_count = kMaxPrintedGapTraceRows / 2;
    const std::size_t suffix_count = kMaxPrintedGapTraceRows - prefix_count;
    int previous_step = 0;
    for (std::size_t i = 0; i < prefix_count; ++i) {
        print_gap_trace_row(graph, trace[i], previous_step);
        previous_step = trace[i].step;
    }
    const std::size_t skipped = trace.size() - prefix_count - suffix_count;
    std::cout << "  ... skipped " << skipped << " intermediate entries ...\n";
    previous_step = trace[trace.size() - suffix_count - 1].step;
    for (std::size_t i = trace.size() - suffix_count; i < trace.size(); ++i) {
        print_gap_trace_row(graph, trace[i], previous_step);
        previous_step = trace[i].step;
    }
}

void print_mass_metrics(
    const std::string &prefix,
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
) {
    const auto protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value() || scores.empty()) {
        return;
    }
    std::cout << prefix << " protected group: " << *protected_group << "\n";
    std::cout << prefix << " Katz mass share: " << std::fixed << std::setprecision(8)
              << katz_mass_share(graph, scores, target_counts) << "\n";
    std::cout << prefix << " Katz mass target: " << std::fixed << std::setprecision(8)
              << protected_group_target_share(target_counts) << "\n";
    std::cout << prefix << " Katz mass objective: " << std::scientific << std::setprecision(12)
              << katz_mass_objective(graph, scores, target_counts) << "\n";
    std::cout << std::fixed;
}

struct GapIncreaseSummary {
    bool increased = false;
    int count = 0;
    double max_delta = 0.0;
    int first_from_step = 0;
    int first_to_step = 0;
    double first_from_gap = 0.0;
    double first_to_gap = 0.0;
};

GapIncreaseSummary summarize_gap_increases(
    const std::vector<GapTraceEntry> &trace
) {
    GapIncreaseSummary summary;
    if (trace.size() < 2) {
        return summary;
    }
    for (std::size_t i = 1; i < trace.size(); ++i) {
        const auto &previous = trace[i - 1];
        const auto &current = trace[i];
        const double delta = current.gap - previous.gap;
        if (delta <= kEps) {
            continue;
        }
        if (!summary.increased) {
            summary.increased = true;
            summary.first_from_step = previous.step;
            summary.first_to_step = current.step;
            summary.first_from_gap = previous.gap;
            summary.first_to_gap = current.gap;
        }
        ++summary.count;
        summary.max_delta = std::max(summary.max_delta, delta);
    }
    return summary;
}

double mean_absolute_score_change(
    const std::vector<double> &initial_scores,
    const std::vector<double> &final_scores
) {
    if (initial_scores.empty() || initial_scores.size() != final_scores.size()) {
        return -1.0;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < initial_scores.size(); ++i) {
        total += std::abs(final_scores[i] - initial_scores[i]);
    }
    return total / static_cast<double>(initial_scores.size());
}

double relative_mean_absolute_score_change(
    const std::vector<double> &initial_scores,
    const std::vector<double> &final_scores
) {
    if (initial_scores.empty() || initial_scores.size() != final_scores.size()) {
        return -1.0;
    }
    double total_change = 0.0;
    double total_initial = 0.0;
    for (std::size_t i = 0; i < initial_scores.size(); ++i) {
        total_change += std::abs(final_scores[i] - initial_scores[i]);
        total_initial += std::abs(initial_scores[i]);
    }
    if (total_initial <= 0.0) {
        return -1.0;
    }
    return total_change / total_initial;
}

MaxScoreChange max_absolute_score_change(
    const std::vector<double> &initial_scores,
    const std::vector<double> &final_scores
) {
    if (initial_scores.empty() || initial_scores.size() != final_scores.size()) {
        return {};
    }
    MaxScoreChange best;
    best.value = 0.0;
    for (std::size_t i = 0; i < initial_scores.size(); ++i) {
        const double change = std::abs(final_scores[i] - initial_scores[i]);
        if (change > best.value) {
            best.index = i;
            best.value = change;
        }
    }
    return best;
}

std::vector<std::size_t> score_ranks(const std::vector<double> &scores) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (scores[lhs] != scores[rhs]) {
            return scores[lhs] > scores[rhs];
        }
        return lhs < rhs;
    });

    std::vector<std::size_t> ranks(scores.size(), 0);
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        ranks[order[rank]] = rank;
    }
    return ranks;
}

double pearson_correlation(
    const std::vector<double> &initial_scores,
    const std::vector<double> &final_scores
) {
    const std::size_t n = initial_scores.size();
    if (n == 0 || n != final_scores.size()) {
        return -2.0;
    }

    long double initial_sum = 0.0L;
    long double final_sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        initial_sum += initial_scores[i];
        final_sum += final_scores[i];
    }

    const long double initial_mean = initial_sum / static_cast<long double>(n);
    const long double final_mean = final_sum / static_cast<long double>(n);
    long double numerator = 0.0L;
    long double initial_sq = 0.0L;
    long double final_sq = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const long double initial_delta = initial_scores[i] - initial_mean;
        const long double final_delta = final_scores[i] - final_mean;
        numerator += initial_delta * final_delta;
        initial_sq += initial_delta * initial_delta;
        final_sq += final_delta * final_delta;
    }

    const long double denominator = std::sqrt(initial_sq * final_sq);
    if (denominator <= 0.0L) {
        return n <= 1 ? 1.0 : -2.0;
    }
    return static_cast<double>(numerator / denominator);
}

double full_graph_spearman(
    const std::vector<double> &initial_scores,
    const std::vector<double> &final_scores
) {
    const std::size_t n = initial_scores.size();
    if (n == 0 || n != final_scores.size()) {
        return -2.0;
    }
    if (n <= 1) {
        return 1.0;
    }

    const auto initial_ranks = score_ranks(initial_scores);
    const auto final_ranks = score_ranks(final_scores);
    long double total = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const long double diff = static_cast<long double>(initial_ranks[i]) -
                                 static_cast<long double>(final_ranks[i]);
        total += diff * diff;
    }
    const long double count = static_cast<long double>(n);
    const long double denominator = count * (count * count - 1.0L);
    return static_cast<double>(1.0L - (6.0L * total) / denominator);
}

}  // namespace

void print_usage() {
    std::cout
        << "Usage: top_k_runner --dataset <name> --k <k> --target male:3,female:3 [options]\n"
        << "Pipeline:\n"
        << "  runner -> resolve alpha -> compute Katz centrality -> run fair algorithm -> print summary\n"
        << "Options:\n"
        << "  --dataset <Blogs|Hopkins|Retweet|Deezer|Penn|Pokec|bpa|custom>\n"
        << "  --backend <eigen_direct>\n"
        << "  --algorithm <gap_greedy|katz_mass|optimal|dense|blade|blade_no_batch|same_group_support>\n"
        << "  --update-mode <jacobi|sherman_morrison>\n"
        << "  --k <int>\n"
        << "  --budget <int>       edit budget; -1 means unbounded for all top-k algorithms\n"
        << "  --target male:3,female:3\n"
        << "  --epsilon <float>      unfairness tolerance; default 0\n"
        << "  --data-root <path>\n"
        << "  --bpa-size <int> --bpa-rho <float>   (BPA edge files are directed u->v; loaded as directed)\n"
        << "  --edge-path <path> --group-path <path>   (custom only)\n"
        << "  --directed | --undirected                (public dataset names/custom)\n"
        << "  --edge-admissibility <any|two_hop>  addable edge filter for graph-design algorithms; default any\n"
        << "  --frontier-limit <int>   dense frontier limit, blade q, or GapGreedy q; 0 means dense full frontier or default q="
        << kDefaultBladeFrontierLimit << "\n"
        << "  --gap-greedy-frontier-limit <int>  GapGreedy boundary candidate count q; overrides --frontier-limit when positive; default "
        << kDefaultGapGreedyFrontierLimit << "\n"
        << "  --same-group-window <int>  SameGroupSupport outside-top-k same-group support window; default 100\n"
        << "  --katz-mass-attempts-per-commit <int>  KatzMass sampled edge attempts before each score update; default 1\n"
        << "  --katz-mass-attempt-fraction <float>  KatzMass attempts per commit as a fraction of initial edges\n"
        << "  --katz-mass-max-commits <int>  optional KatzMass score update commit limit; omitted or 0 means unlimited\n"
        << "  --alpha <float>\n"
        << "  --seed <int>\n";
}

ExperimentOptions parse_args(int argc, char **argv) {
    ExperimentOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--dataset") {
            options.dataset = require_value(arg);
        } else if (arg == "--backend") {
            options.backend = require_value(arg);
        } else if (arg == "--algorithm") {
            options.algorithm = require_value(arg);
        } else if (arg == "--update-mode") {
            options.update_mode = require_value(arg);
        } else if (arg == "--k") {
            options.k = std::stoi(require_value(arg));
        } else if (arg == "--budget") {
            options.budget = std::stoi(require_value(arg));
        } else if (arg == "--target") {
            options.target_counts = parse_target_counts(require_value(arg));
        } else if (arg == "--epsilon") {
            options.epsilon = std::stod(require_value(arg));
        } else if (arg == "--data-root") {
            options.data_root = require_value(arg);
        } else if (arg == "--bpa-size") {
            options.bpa_size = std::stoi(require_value(arg));
        } else if (arg == "--bpa-rho") {
            options.bpa_rho = require_value(arg);
        } else if (arg == "--edge-path") {
            options.edge_path = require_value(arg);
        } else if (arg == "--group-path") {
            options.group_path = require_value(arg);
        } else if (arg == "--directed") {
            options.directed = true;
        } else if (arg == "--undirected") {
            options.directed = false;
        } else if (arg == "--edge-admissibility") {
            options.edge_admissibility = canonicalize_token(require_value(arg));
        } else if (arg == "--frontier-limit") {
            options.frontier_limit = std::stoi(require_value(arg));
        } else if (arg == "--gap-greedy-frontier-limit") {
            options.gap_greedy_frontier_limit = std::stoi(require_value(arg));
        } else if (arg == "--same-group-window") {
            options.same_group_window = std::stoi(require_value(arg));
        } else if (arg == "--katz-mass-attempts-per-commit") {
            options.katz_mass_attempts_per_commit = std::stoi(require_value(arg));
        } else if (arg == "--katz-mass-attempt-fraction") {
            options.katz_mass_attempt_fraction = std::stod(require_value(arg));
        } else if (arg == "--katz-mass-max-commits") {
            options.katz_mass_max_commits = std::stoi(require_value(arg));
        } else if (arg == "--alpha") {
            options.alpha = std::stod(require_value(arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(std::stoul(require_value(arg)));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.dataset.empty()) {
        throw std::runtime_error("--dataset is required.");
    }
    if (options.k <= 0) {
        throw std::runtime_error("--k is required.");
    }
    if (options.target_counts.empty()) {
        throw std::runtime_error("--target is required.");
    }
    if (options.frontier_limit < 0) {
        throw std::runtime_error("--frontier-limit must be non-negative.");
    }
    if (options.gap_greedy_frontier_limit < 0) {
        throw std::runtime_error("--gap-greedy-frontier-limit must be non-negative.");
    }
    if (options.same_group_window <= 0) {
        throw std::runtime_error("--same-group-window must be positive.");
    }
    if (options.katz_mass_attempts_per_commit <= 0) {
        throw std::runtime_error("--katz-mass-attempts-per-commit must be positive.");
    }
    if (options.katz_mass_attempt_fraction < 0.0) {
        throw std::runtime_error("--katz-mass-attempt-fraction must be non-negative.");
    }
    if (options.katz_mass_max_commits.has_value()) {
        if (*options.katz_mass_max_commits < 0) {
            throw std::runtime_error("--katz-mass-max-commits must be non-negative.");
        }
        if (*options.katz_mass_max_commits == 0) {
            options.katz_mass_max_commits.reset();
        }
    }
    if (options.edge_admissibility == "twohop") {
        options.edge_admissibility = "two_hop";
    }
    if (options.edge_admissibility != "any" &&
        options.edge_admissibility != "two_hop") {
        throw std::runtime_error("--edge-admissibility must be any or two_hop.");
    }
    if (options.epsilon < 0.0) {
        throw std::runtime_error("--epsilon must be non-negative.");
    }
    const AlgorithmSelection algorithm_selection = resolve_algorithm_selection(options.algorithm);
    const bool unbounded_budget_allowed =
        options.budget == -1 &&
        (algorithm_selection.kind == FairAlgorithmKind::GapGreedy ||
         algorithm_selection.kind == FairAlgorithmKind::KatzMass ||
         algorithm_selection.kind == FairAlgorithmKind::Optimal ||
         algorithm_selection.kind == FairAlgorithmKind::Dense ||
         algorithm_selection.kind == FairAlgorithmKind::Blade ||
         algorithm_selection.kind == FairAlgorithmKind::BladeNoBatch ||
         algorithm_selection.kind == FairAlgorithmKind::SameGroupSupport);
    if (options.budget < 0 && !unbounded_budget_allowed) {
        throw std::runtime_error(
            "--budget must be non-negative; -1 is supported for gap_greedy, "
            "katz_mass, optimal, dense, blade, blade_no_batch, and same_group_support."
        );
    }
    (void)resolve_backend_selection(options.backend);
    (void)resolve_update_mode(options.update_mode, algorithm_selection.kind);
    const int target_sum = sum_target_counts(options.target_counts);
    if (target_sum != options.k) {
        throw std::runtime_error("Target counts must sum to k.");
    }

    return options;
}

void print_summary(const Graph &graph, const ExperimentReport &report) {
    std::cout << "====================== Experiment Summary ======================\n";
    std::cout << "dataset: " << report.dataset.name << "\n";
    std::cout << "graph: nodes=" << graph.nodes.size()
              << ", edges=" << report.initial_edge_count
              << ", directed=" << (graph.directed ? "true" : "false") << "\n";
    std::cout << "backend: " << report.initial.centrality.backend.resolved << "\n";
    std::cout << "algorithm: " << report.fair_result.algorithm_resolved << "\n";
    std::cout << "update mode: " << report.fair_result.update_mode_resolved << "\n";
    const FairAlgorithmKind algorithm_kind =
        resolve_algorithm_selection(report.options.algorithm).kind;
    if (algorithm_kind == FairAlgorithmKind::Blade ||
        algorithm_kind == FairAlgorithmKind::BladeNoBatch) {
        const int blade_q = report.options.frontier_limit > 0
            ? report.options.frontier_limit
            : kDefaultBladeFrontierLimit;
        std::cout << "frontier_limit: " << blade_q << "\n";
        std::cout << "edge admissibility: " << report.options.edge_admissibility << "\n";
    } else if (algorithm_kind == FairAlgorithmKind::Dense) {
        std::cout << "frontier_limit: " << report.options.frontier_limit << "\n";
    } else if (algorithm_kind == FairAlgorithmKind::GapGreedy) {
        const int gap_greedy_q = report.options.gap_greedy_frontier_limit > 0
            ? report.options.gap_greedy_frontier_limit
            : (report.options.frontier_limit > 0
                ? report.options.frontier_limit
                : kDefaultGapGreedyFrontierLimit);
        std::cout << "gap_greedy frontier_limit: " << gap_greedy_q << "\n";
        std::cout << "edge admissibility: " << report.options.edge_admissibility << "\n";
    } else if (algorithm_kind == FairAlgorithmKind::SameGroupSupport) {
        std::cout << "same_group_window: " << report.options.same_group_window << "\n";
        std::cout << "edge admissibility: " << report.options.edge_admissibility << "\n";
    } else if (algorithm_kind == FairAlgorithmKind::KatzMass) {
        std::cout << "baseline: KatzMass directional sampling\n";
        std::cout << "edge admissibility: " << report.options.edge_admissibility << "\n";
    } else if (algorithm_kind == FairAlgorithmKind::Optimal) {
        std::cout << "edge admissibility: " << report.options.edge_admissibility << "\n";
    }
    if (report.options.katz_mass_attempt_fraction > 0.0) {
        std::cout << "katz mass attempt fraction: "
                  << std::fixed << std::setprecision(8)
                  << report.options.katz_mass_attempt_fraction << "\n";
    } else if (report.options.katz_mass_attempts_per_commit != 1) {
        std::cout << "katz mass attempts per commit: "
                  << report.options.katz_mass_attempts_per_commit << "\n";
    }
    if (report.options.katz_mass_max_commits.has_value()) {
        std::cout << "katz mass max commits: "
                  << *report.options.katz_mass_max_commits << "\n";
    }
    std::cout << "epsilon: " << std::fixed << std::setprecision(8) << report.options.epsilon << "\n";
    std::cout << "alpha: " << std::fixed << std::setprecision(8) << report.initial.centrality.alpha << "\n";
    std::cout << "katz centrality seconds: " << std::fixed << std::setprecision(6)
              << report.initial.centrality.elapsed_seconds << "\n";
    std::cout << "fair algorithm seconds: " << std::fixed << std::setprecision(6)
              << report.fair_result.elapsed_seconds << "\n";
    std::cout << "total pipeline seconds: " << std::fixed << std::setprecision(6)
              << report.total_elapsed_seconds << "\n";
    const double score_mae = mean_absolute_score_change(
        report.initial.centrality.scores,
        report.final.centrality.scores
    );
    if (score_mae >= 0.0) {
        std::cout << "score MAE: " << std::scientific << std::setprecision(8)
                  << score_mae << std::fixed << "\n";
    }
    const double relative_score_mae = relative_mean_absolute_score_change(
        report.initial.centrality.scores,
        report.final.centrality.scores
    );
    if (relative_score_mae >= 0.0) {
        std::cout << "relative score MAE: " << std::scientific << std::setprecision(8)
                  << relative_score_mae << std::fixed << "\n";
    }
    const double full_spearman = full_graph_spearman(
        report.initial.centrality.scores,
        report.final.centrality.scores
    );
    if (full_spearman >= -1.0) {
        std::cout << "full graph Spearman: " << std::fixed << std::setprecision(8)
                  << full_spearman << "\n";
    }
    const double pearson = pearson_correlation(
        report.initial.centrality.scores,
        report.final.centrality.scores
    );
    if (pearson >= -1.0) {
        std::cout << "Pearson correlation: " << std::fixed << std::setprecision(8)
                  << pearson << "\n";
    }
    const MaxScoreChange max_score_diff = max_absolute_score_change(
        report.initial.centrality.scores,
        report.final.centrality.scores
    );
    if (max_score_diff.value >= 0.0) {
        std::cout << "max Katz score change: Node " << graph.nodes[max_score_diff.index]
                  << " " << graph.groups[max_score_diff.index]
                  << " (" << std::fixed << std::setprecision(8)
                  << max_score_diff.value << ")\n";
    }
    std::cout << "initial counts: ";
    print_counts(report.initial.counts);
    std::cout << "\n";
    print_mass_metrics(
        "initial",
        graph,
        report.initial.centrality.scores,
        report.options.target_counts
    );
    std::cout << "\ninitial top_k:\n";
    print_top_k_rows(graph, report.initial.top_k_ranking);
    std::cout << "found target: " << (report.fair_result.found ? "true" : "false") << "\n";
    std::cout << "already satisfied: " << (report.fair_result.already_satisfied ? "true" : "false") << "\n";
    std::cout << "candidate attempts: " << report.fair_result.candidate_attempts << "\n";
    const GapIncreaseSummary gap_increases =
        summarize_gap_increases(report.fair_result.gap_trace);
    std::cout << "unfairness increased: "
              << (gap_increases.increased ? "true" : "false") << "\n";
    std::cout << "unfairness increase count: " << gap_increases.count << "\n";
    std::cout << "max unfairness increase: "
              << std::fixed << std::setprecision(8)
              << gap_increases.max_delta << "\n";
    if (gap_increases.increased) {
        std::cout << "first unfairness increase: step "
                  << gap_increases.first_from_step
                  << " gap=" << std::fixed << std::setprecision(8)
                  << gap_increases.first_from_gap
                  << " -> step " << gap_increases.first_to_step
                  << " gap=" << gap_increases.first_to_gap << "\n";
    }
    print_added_edge_summary(graph, report.fair_result.edges);
    print_gap_trace_summary(graph, report.fair_result.gap_trace);
    if (!report.initial.centrality.backend.message.empty()) {
        std::cout << "backend message: " << report.initial.centrality.backend.message << "\n";
    }
    if (!report.fair_result.message.empty()) {
        std::cout << "algorithm message: " << report.fair_result.message << "\n";
    }
    std::cout << "final counts: ";
    print_counts(report.final.counts);
    std::cout << "\n";
    print_mass_metrics(
        "final",
        graph,
        report.final.centrality.scores,
        report.options.target_counts
    );
    std::cout << "\nfinal top_k:\n";
    print_top_k_rows(graph, report.final.top_k_ranking);
}

}  // namespace top_k

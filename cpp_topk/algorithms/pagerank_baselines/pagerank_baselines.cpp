#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Edge {
    int to = 0;
    double weight = 0.0;
    double initial_weight = 0.0;
};

struct Graph {
    std::vector<std::string> node_ids;
    std::vector<int> labels;
    std::vector<std::string> label_names;
    std::vector<std::vector<Edge>> out;
    std::vector<std::vector<std::pair<int, int>>> in;
    std::map<int, int> label_counts;
};

struct Options {
    std::string edge_path;
    std::string group_path;
    std::string algorithm = "pagerank_fairgd";
    int max_iter = 50;
    double gamma = 0.15;
    double lr = 1.0;
    double top_k_fraction = 0.05;
    int top_k = 0;
    int top_k_max = 100;
    double target0 = 0.5;
    bool target0_set = false;
    int walk_number = 20;
    int walk_length = 40;
    double crosswalk_p = 2.0;
    unsigned int seed = 42;
};

struct TopKSnapshot {
    int k = 0;
    std::vector<int> before_counts;
    std::vector<int> after_counts;
    double before_unfairness = 0.0;
    double after_unfairness = 0.0;
};

struct TransitionChangeStats {
    long long support_before = 0;
    long long support_after = 0;
    long long support_added = 0;
    long long support_removed = 0;
    long long support_delta = 0;
    long long support_weight_changed = 0;
    double weight_l1_change = 0.0;
    double weight_l2_change = 0.0;
    double relative_weight_l1_change = 0.0;
    double relative_weight_l2_change = 0.0;
    double max_abs_weight_change = 0.0;
};

using SparseRowMap = std::map<int, double>;

std::string normalize_algorithm_name(const std::string &raw) {
    if (raw == "pagerank_fairgd") {
        return "pagerank_fairgd";
    }
    if (raw == "pagerank_adaptgd") {
        return "pagerank_adaptgd";
    }
    if (raw == "pagerank_fairwalk") {
        return "pagerank_fairwalk";
    }
    if (raw == "pagerank_crosswalk") {
        return "pagerank_crosswalk";
    }
    if (raw == "pagerank_lfprn") {
        return "pagerank_lfprn";
    }
    if (raw == "pagerank_lfpru") {
        return "pagerank_lfpru";
    }
    throw std::runtime_error(
        "--algorithm must be one of: pagerank_fairgd, pagerank_adaptgd, "
        "pagerank_fairwalk, pagerank_crosswalk, pagerank_lfprn, or pagerank_lfpru"
    );
}

std::vector<std::string> sorted_nodes(const std::unordered_map<std::string, int> &seen) {
    std::vector<std::string> nodes;
    nodes.reserve(seen.size());
    for (const auto &[node, _] : seen) {
        nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end(), [](const std::string &a, const std::string &b) {
        try {
            return std::stoll(a) < std::stoll(b);
        } catch (...) {
            return a < b;
        }
    });
    return nodes;
}

bool label_token_less(const std::string &lhs, const std::string &rhs) {
    try {
        const long long left = std::stoll(lhs);
        const long long right = std::stoll(rhs);
        return left < right;
    } catch (...) {
        return lhs < rhs;
    }
}

Graph load_graph(const std::string &edge_path, const std::string &group_path) {
    std::unordered_map<std::string, int> seen;
    std::unordered_map<std::string, std::string> raw_labels;

    std::string u;
    std::string v;
    std::ifstream groups(group_path);
    if (!groups) {
        throw std::runtime_error("failed to open group file: " + group_path);
    }
    std::string label;
    while (groups >> u >> label) {
        seen.emplace(u, static_cast<int>(seen.size()));
        raw_labels[u] = label;
    }

    std::ifstream edges(edge_path);
    if (!edges) {
        throw std::runtime_error("failed to open edge file: " + edge_path);
    }
    while (edges >> u >> v) {
        if (u == v) {
            continue;
        }
        seen.emplace(u, static_cast<int>(seen.size()));
        seen.emplace(v, static_cast<int>(seen.size()));
    }

    std::vector<std::string> label_tokens;
    label_tokens.reserve(raw_labels.size());
    for (const auto &[_, raw_label] : raw_labels) {
        label_tokens.push_back(raw_label);
    }
    std::sort(label_tokens.begin(), label_tokens.end(), label_token_less);
    label_tokens.erase(std::unique(label_tokens.begin(), label_tokens.end()), label_tokens.end());
    std::unordered_map<std::string, int> label_id;
    label_id.reserve(label_tokens.size());
    for (int i = 0; i < static_cast<int>(label_tokens.size()); ++i) {
        label_id[label_tokens[i]] = i;
    }

    Graph graph;
    graph.node_ids = sorted_nodes(seen);
    graph.label_names = label_tokens;
    const int n = static_cast<int>(graph.node_ids.size());
    std::unordered_map<std::string, int> index;
    index.reserve(graph.node_ids.size());
    for (int i = 0; i < n; ++i) {
        index[graph.node_ids[i]] = i;
    }

    graph.labels.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        auto it = raw_labels.find(graph.node_ids[i]);
        if (it != raw_labels.end()) {
            graph.labels[i] = label_id.at(it->second);
        }
        graph.label_counts[graph.labels[i]] += 1;
    }

    std::vector<std::vector<int>> raw_out(n);
    edges.close();
    edges.open(edge_path);
    if (!edges) {
        throw std::runtime_error("failed to reopen edge file: " + edge_path);
    }
    while (edges >> u >> v) {
        if (u == v) {
            continue;
        }
        const auto src = index.find(u);
        const auto dst = index.find(v);
        if (src == index.end() || dst == index.end()) {
            continue;
        }
        raw_out[src->second].push_back(dst->second);
    }
    graph.out.resize(n);
    graph.in.resize(n);
    for (int i = 0; i < n; ++i) {
        auto &row = raw_out[i];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        const double w = row.empty() ? 0.0 : 1.0 / static_cast<double>(row.size());
        for (int dst : row) {
            const int pos = static_cast<int>(graph.out[i].size());
            graph.out[i].push_back(Edge{dst, w, w});
            graph.in[dst].push_back({i, pos});
        }
    }
    return graph;
}

TransitionChangeStats transition_change_stats_from_initial(const Graph &graph) {
    constexpr double eps = 1.0e-12;
    TransitionChangeStats stats;
    double l2_sq = 0.0;
    double before_l1 = 0.0;
    double before_l2_sq = 0.0;

    for (const auto &row : graph.out) {
        SparseRowMap before;
        SparseRowMap after;
        for (const Edge &edge : row) {
            if (std::abs(edge.initial_weight) > eps) {
                before[edge.to] += edge.initial_weight;
            }
            if (std::abs(edge.weight) > eps) {
                after[edge.to] += edge.weight;
            }
        }

        SparseRowMap merged = before;
        for (const auto &[dst, _] : after) {
            merged.try_emplace(dst, 0.0);
        }

        for (const auto &[dst, _] : merged) {
            const auto before_it = before.find(dst);
            const auto after_it = after.find(dst);
            const double initial_weight = before_it == before.end() ? 0.0 : before_it->second;
            const double weight = after_it == after.end() ? 0.0 : after_it->second;
            const bool had_support = std::abs(initial_weight) > eps;
            const bool has_support = std::abs(weight) > eps;
            if (had_support) {
                ++stats.support_before;
                before_l1 += std::abs(initial_weight);
                before_l2_sq += initial_weight * initial_weight;
            }
            if (has_support) {
                ++stats.support_after;
            }
            if (!had_support && has_support) {
                ++stats.support_added;
            } else if (had_support && !has_support) {
                ++stats.support_removed;
            } else if (had_support && has_support && std::abs(weight - initial_weight) > eps) {
                ++stats.support_weight_changed;
            }
            const double diff = weight - initial_weight;
            if (std::abs(diff) > eps) {
                stats.weight_l1_change += std::abs(diff);
                l2_sq += diff * diff;
                stats.max_abs_weight_change = std::max(stats.max_abs_weight_change, std::abs(diff));
            }
        }
    }

    stats.support_delta = stats.support_after - stats.support_before;
    stats.weight_l2_change = std::sqrt(l2_sq);
    const double before_l2 = std::sqrt(before_l2_sq);
    if (before_l1 > 0.0) {
        stats.relative_weight_l1_change = stats.weight_l1_change / before_l1;
    }
    if (before_l2 > 0.0) {
        stats.relative_weight_l2_change = stats.weight_l2_change / before_l2;
    }
    return stats;
}

std::vector<double> target_vector(const Graph &graph, const Options &options) {
    if (graph.label_counts.size() != 2) {
        std::vector<double> target;
        target.reserve(graph.label_counts.size());
        for (const auto &[_, count] : graph.label_counts) {
            target.push_back(static_cast<double>(count) / graph.labels.size());
        }
        return target;
    }
    const double first = options.target0_set ? options.target0 : 0.5;
    return {first, 1.0 - first};
}

void rebuild_in_neighbors(Graph &graph) {
    graph.in.assign(graph.out.size(), {});
    for (int src = 0; src < static_cast<int>(graph.out.size()); ++src) {
        for (int pos = 0; pos < static_cast<int>(graph.out[src].size()); ++pos) {
            graph.in[graph.out[src][pos].to].push_back({src, pos});
        }
    }
}

std::vector<std::string> compressed_labels(Graph &graph) {
    if (!graph.label_names.empty()) {
        return graph.label_names;
    }
    std::vector<std::string> names;
    names.reserve(graph.label_counts.size());
    for (const auto &[label, _] : graph.label_counts) {
        names.push_back(std::to_string(label));
    }
    return names;
}

std::vector<double> pagerank(const Graph &graph, const std::vector<double> &restart, double gamma, int max_iter = 80) {
    const int n = static_cast<int>(graph.labels.size());
    const double alpha = 1.0 - gamma;
    std::vector<double> x(n, 1.0 / n);
    std::vector<double> next(n, 0.0);
    std::vector<int> sinks;
    for (int i = 0; i < n; ++i) {
        if (graph.out[i].empty()) {
            sinks.push_back(i);
        }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        std::fill(next.begin(), next.end(), 0.0);
        double sink_mass = 0.0;
        for (int s : sinks) {
            sink_mass += x[s];
        }
        for (int i = 0; i < n; ++i) {
            for (const Edge &edge : graph.out[i]) {
                next[edge.to] += alpha * x[i] * edge.weight;
            }
        }
        double err = 0.0;
        for (int i = 0; i < n; ++i) {
            next[i] += alpha * sink_mass * restart[i] + gamma * restart[i];
            err += std::abs(next[i] - x[i]);
        }
        x.swap(next);
        if (err < n * 1.0e-10) {
            break;
        }
    }
    return x;
}

std::vector<std::vector<double>> personalized_group_pagerank(
    const Graph &graph,
    const std::vector<std::vector<double>> &group_indicator,
    const std::vector<double> &restart,
    double gamma,
    int steps = 30
) {
    const int g = static_cast<int>(group_indicator.size());
    const int n = static_cast<int>(graph.labels.size());
    const double alpha = 1.0 - gamma;
    std::vector<std::vector<double>> total = group_indicator;
    std::vector<std::vector<double>> current = group_indicator;
    std::vector<int> sinks;
    for (int i = 0; i < n; ++i) {
        if (graph.out[i].empty()) {
            sinks.push_back(i);
        }
    }

    for (int step = 0; step < steps; ++step) {
        std::vector<std::vector<double>> next(g, std::vector<double>(n, 0.0));
        for (int k = 0; k < g; ++k) {
            double restart_dot = 0.0;
            for (int i = 0; i < n; ++i) {
                restart_dot += current[k][i] * restart[i];
                for (const auto &[src, pos] : graph.in[i]) {
                    next[k][src] += alpha * current[k][i] * graph.out[src][pos].weight;
                }
            }
            for (int sink : sinks) {
                next[k][sink] += alpha * restart_dot;
            }
            for (int i = 0; i < n; ++i) {
                total[k][i] += next[k][i];
            }
        }
        current.swap(next);
    }
    return total;
}

std::vector<double> project_simplex(std::vector<double> values) {
    if (values.size() == 1) {
        return {1.0};
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    double sum = 0.0;
    int rho = 0;
    for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
        sum += sorted[i];
        if (sorted[i] - (sum - 1.0) / (i + 1) > 0.0) {
            rho = i + 1;
        }
    }
    double theta_sum = 0.0;
    for (int i = 0; i < rho; ++i) {
        theta_sum += sorted[i];
    }
    const double theta = (theta_sum - 1.0) / rho;
    for (double &value : values) {
        value = std::max(0.0, value - theta);
    }
    return values;
}

std::vector<std::vector<double>> group_indicators(const Graph &graph) {
    const int n = static_cast<int>(graph.labels.size());
    const int g = static_cast<int>(graph.label_counts.size());
    std::vector<std::vector<double>> indicators(g, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        indicators[graph.labels[i]][i] = 1.0;
    }
    return indicators;
}

std::vector<double> group_mass(const Graph &graph, const std::vector<double> &pr) {
    std::vector<double> mass(graph.label_counts.size(), 0.0);
    for (int i = 0; i < static_cast<int>(pr.size()); ++i) {
        mass[graph.labels[i]] += pr[i];
    }
    return mass;
}

double squared_unfairness(const std::vector<double> &actual, const std::vector<double> &target) {
    double loss = 0.0;
    for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
        const double diff = actual[i] - target[i];
        loss += diff * diff;
    }
    return loss / actual.size();
}

double top_k_unfairness(const Graph &graph, const std::vector<double> &scores, int k, const std::vector<double> &target) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::abs(scores[a] - scores[b]) > 1.0e-15) {
            return scores[a] > scores[b];
        }
        return a < b;
    });
    std::vector<double> actual(target.size(), 0.0);
    for (int i = 0; i < k && i < static_cast<int>(order.size()); ++i) {
        actual[graph.labels[order[i]]] += 1.0;
    }
    for (double &value : actual) {
        value /= static_cast<double>(k);
    }
    return squared_unfairness(actual, target);
}

std::vector<int> top_k_counts(const Graph &graph, const std::vector<double> &scores, int k) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::abs(scores[a] - scores[b]) > 1.0e-15) {
            return scores[a] > scores[b];
        }
        return a < b;
    });
    std::vector<int> counts(graph.label_counts.size(), 0);
    for (int i = 0; i < k && i < static_cast<int>(order.size()); ++i) {
        counts[graph.labels[order[i]]] += 1;
    }
    return counts;
}

int effective_top_k_for_nodes(int nodes, const Options &options) {
    int k = options.top_k > 0
        ? options.top_k
        : static_cast<int>(std::ceil(options.top_k_fraction * static_cast<double>(nodes)));

    if (k < 2) {
        k = 2;
    }
    if (k % 2 == 1) {
        ++k;
    }
    if (options.top_k_max > 0 && k > options.top_k_max) {
        k = options.top_k_max;
        if (k % 2 == 1) {
            --k;
        }
    }
    if (k > nodes) {
        k = nodes;
        if (k % 2 == 1) {
            --k;
        }
    }
    if (k < 2) {
        k = 2;
    }
    return k;
}

std::vector<int> distribution_cutoffs(int max_k, int nodes) {
    const std::vector<int> requested = {10, 20, 50, 100, 200, 500};
    std::vector<int> cutoffs;
    cutoffs.reserve(requested.size() + 1);
    for (int k : requested) {
        if (k <= max_k && k <= nodes) {
            cutoffs.push_back(k);
        }
    }
    if (max_k > 0 && max_k <= nodes) {
        cutoffs.push_back(max_k);
    }
    std::sort(cutoffs.begin(), cutoffs.end());
    cutoffs.erase(std::unique(cutoffs.begin(), cutoffs.end()), cutoffs.end());
    return cutoffs;
}

void print_group_counts(
    const std::string &prefix,
    int total,
    const std::vector<int> &counts,
    const std::vector<std::string> &original_labels
) {
    std::cout << prefix << ": total=" << total;
    for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
        const std::string label = i < static_cast<int>(original_labels.size())
            ? original_labels[i]
            : std::to_string(i);
        std::cout << ", label " << label << "=" << counts[i];
    }
    std::cout << "\n";
}

void print_group_mass(
    const std::string &prefix,
    const std::vector<double> &mass,
    const std::vector<std::string> &original_labels
) {
    std::cout << prefix << ": total=1.0000000000";
    for (int i = 0; i < static_cast<int>(mass.size()); ++i) {
        const std::string label = i < static_cast<int>(original_labels.size())
            ? original_labels[i]
            : std::to_string(i);
        std::cout << ", label " << label << "=" << mass[i];
    }
    std::cout << "\n";
}

void print_top_k_distribution_table(
    const std::vector<TopKSnapshot> &snapshots,
    const std::vector<std::string> &original_labels
) {
    std::cout << "Top-K distribution table:\n";
    std::cout << "k\tbefore_unfairness\tafter_unfairness";
    for (const std::string &label : original_labels) {
        std::cout << "\tbefore_count_" << label
                  << "\tafter_count_" << label
                  << "\tbefore_share_" << label
                  << "\tafter_share_" << label;
    }
    std::cout << "\n";
    for (const TopKSnapshot &snapshot : snapshots) {
        std::cout << snapshot.k << "\t"
                  << snapshot.before_unfairness << "\t"
                  << snapshot.after_unfairness;
        for (int i = 0; i < static_cast<int>(original_labels.size()); ++i) {
            const int before_count = i < static_cast<int>(snapshot.before_counts.size())
                ? snapshot.before_counts[i]
                : 0;
            const int after_count = i < static_cast<int>(snapshot.after_counts.size())
                ? snapshot.after_counts[i]
                : 0;
            const double denom = snapshot.k > 0 ? static_cast<double>(snapshot.k) : 1.0;
            std::cout << "\t" << before_count
                      << "\t" << after_count
                      << "\t" << before_count / denom
                      << "\t" << after_count / denom;
        }
        std::cout << "\n";
    }
}

std::vector<std::vector<int>> nodes_by_label(const Graph &graph) {
    std::vector<std::vector<int>> groups(graph.label_counts.size());
    for (int i = 0; i < static_cast<int>(graph.labels.size()); ++i) {
        groups[graph.labels[i]].push_back(i);
    }
    return groups;
}

std::vector<std::vector<double>> row_label_weight_sums(const Graph &graph) {
    std::vector<std::vector<double>> sums(
        graph.out.size(),
        std::vector<double>(graph.label_counts.size(), 0.0)
    );
    for (int src = 0; src < static_cast<int>(graph.out.size()); ++src) {
        for (const Edge &edge : graph.out[src]) {
            sums[src][graph.labels[edge.to]] += edge.weight;
        }
    }
    return sums;
}

std::vector<std::vector<int>> row_label_edge_counts(const Graph &graph) {
    std::vector<std::vector<int>> counts(
        graph.out.size(),
        std::vector<int>(graph.label_counts.size(), 0)
    );
    for (int src = 0; src < static_cast<int>(graph.out.size()); ++src) {
        for (const Edge &edge : graph.out[src]) {
            counts[src][graph.labels[edge.to]] += 1;
        }
    }
    return counts;
}

void run_fairwalk(Graph &graph, const std::vector<double> &target) {
    const auto out_weights = row_label_weight_sums(graph);
    for (int src = 0; src < static_cast<int>(graph.out.size()); ++src) {
        double sum_phi = 0.0;
        for (int label = 0; label < static_cast<int>(target.size()); ++label) {
            if (out_weights[src][label] > 0.0) {
                sum_phi += target[label];
            }
        }
        if (sum_phi <= 0.0) {
            continue;
        }
        for (Edge &edge : graph.out[src]) {
            const int label = graph.labels[edge.to];
            edge.weight = (target[label] / sum_phi) / out_weights[src][label] * edge.weight;
        }
    }
    rebuild_in_neighbors(graph);
}

void run_lfprn(Graph &graph, const std::vector<double> &target) {
    const int n = static_cast<int>(graph.labels.size());
    const int groups = static_cast<int>(graph.label_counts.size());
    const auto label_nodes = nodes_by_label(graph);
    const auto edge_counts = row_label_edge_counts(graph);
    std::vector<std::vector<Edge>> next_out(n);

    for (int src = 0; src < n; ++src) {
        for (int label = 0; label < groups; ++label) {
            if (edge_counts[src][label] == 0) {
                const double weight = target[label] / label_nodes[label].size();
                for (int dst : label_nodes[label]) {
                    next_out[src].push_back(Edge{dst, weight, 0.0});
                }
            } else {
                const double weight = target[label] / edge_counts[src][label];
                for (const Edge &edge : graph.out[src]) {
                    if (graph.labels[edge.to] == label) {
                        next_out[src].push_back(Edge{edge.to, weight, edge.initial_weight});
                    }
                }
            }
        }
    }
    graph.out.swap(next_out);
    rebuild_in_neighbors(graph);
}

void run_lfpru(Graph &graph, const std::vector<double> &target) {
    const int n = static_cast<int>(graph.labels.size());
    const int groups = static_cast<int>(graph.label_counts.size());
    if (groups != 2) {
        throw std::runtime_error("pagerank_lfpru is defined here only for two labels.");
    }
    const auto label_nodes = nodes_by_label(graph);
    const auto edge_counts = row_label_edge_counts(graph);
    std::vector<std::vector<Edge>> next_out(n);

    for (int src = 0; src < n; ++src) {
        if (graph.out[src].empty()) {
            for (int label = 0; label < groups; ++label) {
                const double weight = target[label] / label_nodes[label].size();
                for (int dst : label_nodes[label]) {
                    next_out[src].push_back(Edge{dst, weight, 0.0});
                }
            }
            continue;
        }
        int lower_label = 0;
        for (int label = 0; label < groups; ++label) {
            const double ratio = static_cast<double>(edge_counts[src][label]) /
                                 static_cast<double>(graph.out[src].size());
            if (ratio < target[label]) {
                lower_label = label;
            }
        }
        const int other_label = lower_label == 0 ? 1 : 0;
        const int denom = std::max(edge_counts[src][other_label], 1);
        const double kept = target[other_label] / static_cast<double>(denom);
        std::vector<double> residual(groups, 0.0);
        for (int label = 0; label < groups; ++label) {
            residual[label] = target[label] - kept * edge_counts[src][label];
        }
        for (const Edge &edge : graph.out[src]) {
            next_out[src].push_back(Edge{edge.to, kept, edge.initial_weight});
        }
        for (int label = 0; label < groups; ++label) {
            if (std::abs(residual[label]) <= 1.0e-15) {
                continue;
            }
            const double weight = residual[label] / label_nodes[label].size();
            for (int dst : label_nodes[label]) {
                next_out[src].push_back(Edge{dst, weight, 0.0});
            }
        }
        std::sort(next_out[src].begin(), next_out[src].end(), [](const Edge &a, const Edge &b) {
            return a.to < b.to;
        });
        std::vector<Edge> merged;
        for (const Edge &edge : next_out[src]) {
            if (!merged.empty() && merged.back().to == edge.to) {
                merged.back().weight += edge.weight;
                merged.back().initial_weight += edge.initial_weight;
            } else {
                merged.push_back(edge);
            }
        }
        next_out[src].swap(merged);
    }
    graph.out.swap(next_out);
    rebuild_in_neighbors(graph);
}

int sample_edge_index(const std::vector<Edge> &row, std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double draw = dist(rng);
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(row.size()); ++i) {
        cumulative += row[i].weight;
        if (draw <= cumulative) {
            return i;
        }
    }
    return static_cast<int>(row.size()) - 1;
}

std::vector<double> crosswalk_node_m(const Graph &graph, const Options &options) {
    const int n = static_cast<int>(graph.labels.size());
    std::vector<double> m(n, 0.0);
    std::mt19937 rng(options.seed);
    for (int start = 0; start < n; ++start) {
        int cnt_other = 0;
        int cnt_total = 0;
        for (int walk_id = 0; walk_id < options.walk_number; ++walk_id) {
            int current = start;
            for (int step = 0; step < options.walk_length; ++step) {
                ++cnt_total;
                if (graph.labels[current] != graph.labels[start]) {
                    ++cnt_other;
                }
                if (graph.out[current].empty()) {
                    break;
                }
                current = graph.out[current][sample_edge_index(graph.out[current], rng)].to;
            }
        }
        m[start] = cnt_total == 0 ? 0.0 : static_cast<double>(cnt_other) / cnt_total;
    }
    return m;
}

void run_crosswalk(Graph &graph, const Options &options, const std::vector<double> &target) {
    const int groups = static_cast<int>(graph.label_counts.size());
    const auto out_weights = row_label_weight_sums(graph);
    const auto m = crosswalk_node_m(graph, options);
    for (int src = 0; src < static_cast<int>(graph.out.size()); ++src) {
        std::vector<double> label_weights(groups, 0.0);
        for (const Edge &edge : graph.out[src]) {
            label_weights[graph.labels[edge.to]] += edge.weight * std::pow(m[edge.to], options.crosswalk_p);
        }
        std::vector<double> next_weights(graph.out[src].size(), 0.0);
        double norm = 0.0;
        for (int pos = 0; pos < static_cast<int>(graph.out[src].size()); ++pos) {
            const Edge &edge = graph.out[src][pos];
            const int label = graph.labels[edge.to];
            if (label_weights[label] == 0.0) {
                next_weights[pos] = out_weights[src][label] == 0.0
                    ? 0.0
                    : (edge.weight / out_weights[src][label]) * target[label];
            } else {
                next_weights[pos] = ((edge.weight * std::pow(m[edge.to], options.crosswalk_p)) /
                                     label_weights[label]) * target[label];
            }
            norm += next_weights[pos];
        }
        if (norm > 0.0) {
            for (int pos = 0; pos < static_cast<int>(graph.out[src].size()); ++pos) {
                graph.out[src][pos].weight = next_weights[pos] / norm;
            }
        }
    }
    rebuild_in_neighbors(graph);
}

void project_rows(Graph &graph) {
    for (auto &row : graph.out) {
        if (row.empty()) {
            continue;
        }
        std::vector<double> weights;
        weights.reserve(row.size());
        for (const Edge &edge : row) {
            weights.push_back(edge.weight);
        }
        weights = project_simplex(std::move(weights));
        for (int i = 0; i < static_cast<int>(row.size()); ++i) {
            row[i].weight = weights[i];
        }
    }
}

void run_fairgd(Graph &graph, const Options &options, const std::vector<double> &target) {
    const int n = static_cast<int>(graph.labels.size());
    const int groups = static_cast<int>(graph.label_counts.size());
    const auto indicators = group_indicators(graph);
    const std::vector<double> restart(n, 1.0 / n);
    const double step_size = options.lr * 2.0 * (1.0 - options.gamma) / groups;

    for (int iter = 0; iter < options.max_iter; ++iter) {
        const auto pr = pagerank(graph, restart, options.gamma);
        const auto mass = group_mass(graph, pr);
        std::vector<double> diff(groups, 0.0);
        for (int g = 0; g < groups; ++g) {
            diff[g] = target[g] - mass[g];
        }
        const auto one_u = personalized_group_pagerank(graph, indicators, restart, options.gamma);
        for (int src = 0; src < n; ++src) {
            for (Edge &edge : graph.out[src]) {
                double group_term = 0.0;
                for (int g = 0; g < groups; ++g) {
                    group_term += diff[g] * one_u[g][edge.to];
                }
                edge.weight += step_size * pr[src] * group_term;
            }
        }
        project_rows(graph);
    }
}

void run_adaptgd(Graph &graph, const Options &options, const std::vector<double> &target) {
    const int n = static_cast<int>(graph.labels.size());
    const int groups = static_cast<int>(graph.label_counts.size());
    const auto indicators = group_indicators(graph);
    std::vector<std::vector<double>> restarts(groups, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        restarts[graph.labels[i]][i] = 1.0 / graph.label_counts.at(graph.labels[i]);
    }
    const double step_size = options.lr * 2.0 * (1.0 - options.gamma) / (groups * groups);

    for (int iter = 0; iter < options.max_iter; ++iter) {
        std::vector<std::vector<double>> delta(n);
        for (int l = 0; l < groups; ++l) {
            const auto pr = pagerank(graph, restarts[l], options.gamma);
            const auto one_u = personalized_group_pagerank(graph, indicators, restarts[l], options.gamma);
            const auto mass = group_mass(graph, pr);
            for (int src = 0; src < n; ++src) {
                if (delta[src].empty()) {
                    delta[src].assign(graph.out[src].size(), 0.0);
                }
                for (int pos = 0; pos < static_cast<int>(graph.out[src].size()); ++pos) {
                    const int dst = graph.out[src][pos].to;
                    for (int k = 0; k < groups; ++k) {
                        const double this_loss = mass[k] - target[k];
                        delta[src][pos] -= step_size * pr[src] * one_u[k][dst] * this_loss;
                    }
                }
            }
        }
        for (int src = 0; src < n; ++src) {
            for (int pos = 0; pos < static_cast<int>(graph.out[src].size()); ++pos) {
                graph.out[src][pos].weight += delta[src][pos];
            }
        }
        project_rows(graph);
    }
}

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const std::string &name) {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--edge-path") {
            options.edge_path = value(arg);
        } else if (arg == "--group-path") {
            options.group_path = value(arg);
        } else if (arg == "--algorithm") {
            options.algorithm = normalize_algorithm_name(value(arg));
        } else if (arg == "--max-iter") {
            options.max_iter = std::stoi(value(arg));
        } else if (arg == "--lr") {
            options.lr = std::stod(value(arg));
        } else if (arg == "--gamma") {
            options.gamma = std::stod(value(arg));
        } else if (arg == "--top-k-fraction") {
            options.top_k_fraction = std::stod(value(arg));
        } else if (arg == "--top-k") {
            options.top_k = std::stoi(value(arg));
        } else if (arg == "--top-k-max") {
            options.top_k_max = std::stoi(value(arg));
        } else if (arg == "--target0") {
            options.target0 = std::stod(value(arg));
            options.target0_set = true;
        } else if (arg == "--walk-number") {
            options.walk_number = std::stoi(value(arg));
        } else if (arg == "--walk-length") {
            options.walk_length = std::stoi(value(arg));
        } else if (arg == "--crosswalk-p") {
            options.crosswalk_p = std::stod(value(arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(std::stoul(value(arg)));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.edge_path.empty() || options.group_path.empty()) {
        throw std::runtime_error("--edge-path and --group-path are required");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("--top-k must be non-negative");
    }
    if (options.top_k_fraction <= 0.0 && options.top_k == 0) {
        throw std::runtime_error("--top-k-fraction must be positive when --top-k is not set");
    }
    if (options.top_k_max < 0) {
        throw std::runtime_error("--top-k-max must be non-negative");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_args(argc, argv);
        Graph graph = load_graph(options.edge_path, options.group_path);
        const auto original_labels = compressed_labels(graph);
        const auto target = target_vector(graph, options);
        const int n = static_cast<int>(graph.labels.size());
        const int k = effective_top_k_for_nodes(n, options);
        const std::vector<double> restart(n, 1.0 / n);

        const auto before_pr = pagerank(graph, restart, options.gamma);
        const auto before_mass = group_mass(graph, before_pr);
        const double before_whole = squared_unfairness(before_mass, target);
        const double before_top_k = top_k_unfairness(graph, before_pr, k, target);
        const auto before_top_k_counts = top_k_counts(graph, before_pr, k);

        const auto algorithm_started = std::chrono::steady_clock::now();
        if (options.algorithm == "pagerank_fairgd") {
            run_fairgd(graph, options, target);
        } else if (options.algorithm == "pagerank_adaptgd") {
            run_adaptgd(graph, options, target);
        } else if (options.algorithm == "pagerank_fairwalk") {
            run_fairwalk(graph, target);
        } else if (options.algorithm == "pagerank_crosswalk") {
            run_crosswalk(graph, options, target);
        } else if (options.algorithm == "pagerank_lfprn") {
            run_lfprn(graph, target);
        } else if (options.algorithm == "pagerank_lfpru") {
            run_lfpru(graph, target);
        } else {
            throw std::runtime_error(
                "--algorithm must be pagerank_fairgd, pagerank_adaptgd, "
                "pagerank_fairwalk, pagerank_crosswalk, pagerank_lfprn, or pagerank_lfpru"
            );
        }
        const auto algorithm_finished = std::chrono::steady_clock::now();
        const double runtime_seconds =
            std::chrono::duration<double>(algorithm_finished - algorithm_started).count();
        const TransitionChangeStats change_stats =
            transition_change_stats_from_initial(graph);

        const auto after_pr = pagerank(graph, restart, options.gamma);
        const auto after_mass = group_mass(graph, after_pr);
        const double after_whole = squared_unfairness(after_mass, target);
        const double after_top_k = top_k_unfairness(graph, after_pr, k, target);
        const auto after_top_k_counts = top_k_counts(graph, after_pr, k);

        std::vector<TopKSnapshot> top_k_snapshots;
        for (int cutoff : distribution_cutoffs(k, n)) {
            top_k_snapshots.push_back(TopKSnapshot{
                cutoff,
                top_k_counts(graph, before_pr, cutoff),
                top_k_counts(graph, after_pr, cutoff),
                top_k_unfairness(graph, before_pr, cutoff, target),
                top_k_unfairness(graph, after_pr, cutoff, target)
            });
        }

        std::vector<int> whole_counts(graph.label_counts.size(), 0);
        for (const auto &[label, count] : graph.label_counts) {
            whole_counts[label] = count;
        }

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "algorithm: " << options.algorithm << "\n";
        std::cout << "nodes: " << n << "\n";
        std::cout << "top_k_fraction: " << options.top_k_fraction << "\n";
        std::cout << "top_k_max: " << options.top_k_max << "\n";
        std::cout << "top_k: " << k << "\n";
        std::cout << "labels:";
        for (int i = 0; i < static_cast<int>(original_labels.size()); ++i) {
            std::cout << " " << original_labels[i] << "->" << i;
        }
        std::cout << "\n";
        std::cout << "target:";
        for (double value : target) {
            std::cout << " " << value;
        }
        std::cout << "\n";
        print_group_counts("Whole graph counts", n, whole_counts, original_labels);
        print_group_mass("Before whole graph pagerank mass", before_mass, original_labels);
        print_group_mass("After whole graph pagerank mass", after_mass, original_labels);
        print_group_counts("Before Top-K counts", k, before_top_k_counts, original_labels);
        print_group_counts("After Top-K counts", k, after_top_k_counts, original_labels);
        std::cout << "Before whole graph unfairness: " << before_whole << "\n";
        std::cout << "After whole graph unfairness: " << after_whole << "\n";
        std::cout << "Before Top-K unfairness: " << before_top_k << "\n";
        std::cout << "After Top-K unfairness: " << after_top_k << "\n";
        print_top_k_distribution_table(top_k_snapshots, original_labels);
        std::cout << "runtime_seconds: " << runtime_seconds << "\n";
        std::cout << "support before: " << change_stats.support_before << "\n";
        std::cout << "support after: " << change_stats.support_after << "\n";
        std::cout << "support added: " << change_stats.support_added << "\n";
        std::cout << "support removed: " << change_stats.support_removed << "\n";
        std::cout << "support delta: " << change_stats.support_delta << "\n";
        std::cout << "support weight changed: " << change_stats.support_weight_changed << "\n";
        std::cout << "weight L1 change: " << change_stats.weight_l1_change << "\n";
        std::cout << "weight L2 change: " << change_stats.weight_l2_change << "\n";
        std::cout << "relative weight L1 change: " << change_stats.relative_weight_l1_change << "\n";
        std::cout << "relative weight L2 change: " << change_stats.relative_weight_l2_change << "\n";
        std::cout << "max abs weight change: " << change_stats.max_abs_weight_change << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}

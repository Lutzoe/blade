#include "src/types.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <regex>
#include <set>
#include <stdexcept>

namespace top_k {
namespace {

bool has_numeric_suffix(const std::string &value, std::string &prefix, long long &number) {
    static const std::regex pattern(R"(^(.*?)(\d+)$)");
    std::smatch match;
    if (!std::regex_match(value, match, pattern)) {
        return false;
    }
    prefix = match[1].str();
    number = std::stoll(match[2].str());
    return true;
}

bool node_less(const std::string &lhs, const std::string &rhs) {
    std::string lp;
    std::string rp;
    long long ln = 0;
    long long rn = 0;
    const bool lhs_num = has_numeric_suffix(lhs, lp, ln);
    const bool rhs_num = has_numeric_suffix(rhs, rp, rn);
    if (lhs_num != rhs_num) {
        return lhs_num > rhs_num;
    }
    if (lhs_num && rhs_num) {
        if (lp != rp) {
            return lp < rp;
        }
        if (ln != rn) {
            return ln < rn;
        }
    }
    return lhs < rhs;
}

bool ranked_node_better(const RankedNode &lhs, const RankedNode &rhs) {
    if (std::abs(lhs.score - rhs.score) > kEps) {
        return lhs.score > rhs.score;
    }
    return lhs.index < rhs.index;
}

std::string join_path(const std::string &lhs, const std::string &rhs) {
    if (rhs.empty()) {
        return lhs;
    }
    if (rhs.front() == '/') {
        return rhs;
    }
    if (!lhs.empty() && lhs.back() == '/') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

std::unordered_map<std::string, std::string> load_group_labels(const std::string &path) {
    std::unordered_map<std::string, std::string> labels;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open group file: " + path);
    }
    std::string node;
    std::string group;
    while (input >> node >> group) {
        labels[node] = group;
    }
    return labels;
}

bool is_bpa_dataset_name(const std::string &name) {
    return name == "bpa" || name == "bpa1" || name == "bpa2" ||
           name == "bpa3" || name == "bpa4";
}

bool is_public_real_dataset_name(const std::string &name) {
    return name == "Blogs" || name == "Hopkins" || name == "Retweet" ||
           name == "Deezer" || name == "Penn" || name == "Twitch" ||
           name == "DBLP" || name == "Pokec";
}

bool default_directed_for_public_dataset(const std::string &name) {
    return name == "Blogs" || name == "Retweet" || name == "Pokec";
}

std::string public_dataset_data_dir(const std::string &name) {
    return name;
}

std::string public_dataset_report_name(const std::string &name) {
    return name;
}

}  // namespace

DatasetSpec resolve_dataset(const ExperimentOptions &options) {
    const std::string name = options.dataset;
    if (is_bpa_dataset_name(name)) {
        if (!options.bpa_size.has_value() || !options.bpa_rho.has_value()) {
            throw std::runtime_error(name + " requires --bpa-size and --bpa-rho.");
        }
        if (options.edge_path.has_value() || options.group_path.has_value()) {
            if (!options.edge_path.has_value() || !options.group_path.has_value()) {
                throw std::runtime_error(name + " requires both --edge-path and --group-path when either override is set.");
            }
            return DatasetSpec{
                name,
                *options.edge_path,
                *options.group_path,
                true,
            };
        }
        std::string prefix = "bpa_" + std::to_string(*options.bpa_size);
        if (name != "bpa") {
            std::string rho_label = *options.bpa_rho;
            std::replace(rho_label.begin(), rho_label.end(), '.', 'p');
            prefix += "_rho_" + rho_label;
        }
        return DatasetSpec{
            name,
            join_path(join_path(options.data_root, name), prefix + ".txt"),
            join_path(join_path(options.data_root, name), prefix + "_gender.txt"),
            true,
        };
    }
    if (is_public_real_dataset_name(name)) {
        if (options.edge_path.has_value() || options.group_path.has_value()) {
            throw std::runtime_error(name + " uses DATA_ROOT/" + public_dataset_data_dir(name) + "/edges.txt and groups.txt; use --dataset custom for explicit paths.");
        }
        const std::string data_dir = public_dataset_data_dir(name);
        return DatasetSpec{
            public_dataset_report_name(name),
            join_path(join_path(options.data_root, data_dir), "edges.txt"),
            join_path(join_path(options.data_root, data_dir), "groups.txt"),
            options.directed.value_or(default_directed_for_public_dataset(name)),
        };
    }
    if (name == "custom") {
        if (!options.edge_path.has_value() || !options.group_path.has_value()) {
            throw std::runtime_error("custom requires --edge-path and --group-path.");
        }
        return DatasetSpec{
            "custom",
            *options.edge_path,
            *options.group_path,
            options.directed.value_or(false),
        };
    }
    throw std::runtime_error("dataset must be one of: Blogs, Hopkins, Retweet, Deezer, Penn, Twitch, DBLP, Pokec, bpa, bpa1, bpa2, bpa3, bpa4, custom");
}

Graph load_graph(const DatasetSpec &spec) {
    std::ifstream input(spec.edge_path);
    if (!input) {
        throw std::runtime_error("Failed to open edge file: " + spec.edge_path);
    }
    std::vector<std::string> node_ids;
    std::unordered_map<std::string, std::size_t> ingest_index;
    std::vector<std::pair<std::size_t, std::size_t>> raw_edges;
    auto ingest_node = [&](const std::string &node) {
        const auto [it, inserted] = ingest_index.emplace(node, node_ids.size());
        if (inserted) {
            node_ids.push_back(node);
        }
        return it->second;
    };

    std::string u;
    std::string v;
    while (input >> u >> v) {
        if (u == v) {
            continue;
        }
        const std::size_t iu = ingest_node(u);
        const std::size_t iv = ingest_node(v);
        raw_edges.push_back({iu, iv});
        if (!spec.directed) {
            raw_edges.push_back({iv, iu});
        }
    }

    const auto labels = load_group_labels(spec.group_path);
    for (const auto &[node, _] : labels) {
        ingest_node(node);
    }

    Graph graph;
    graph.directed = spec.directed;
    graph.nodes = node_ids;
    std::sort(graph.nodes.begin(), graph.nodes.end(), node_less);
    graph.groups.resize(graph.nodes.size(), "unknown");
    graph.out_neighbors.assign(graph.nodes.size(), {});
    graph.in_neighbors.assign(graph.nodes.size(), {});
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        graph.index_of[graph.nodes[i]] = i;
    }

    std::vector<std::size_t> remap(node_ids.size(), 0);
    for (std::size_t i = 0; i < node_ids.size(); ++i) {
        remap[i] = graph.index_of.at(node_ids[i]);
    }

    for (const auto &[node, group] : labels) {
        const auto it = graph.index_of.find(node);
        if (it != graph.index_of.end()) {
            graph.groups[it->second] = group;
        }
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    edges.reserve(raw_edges.size());
    for (const auto &[src, dst] : raw_edges) {
        edges.push_back({remap[src], remap[dst]});
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    for (const auto &[src, dst] : edges) {
        graph.add_edge(src, dst);
    }
    return graph;
}

double resolve_alpha(const Graph &graph, const std::string &dataset_name) {
    const int dmax = is_bpa_dataset_name(dataset_name)
        ? graph.max_in_degree()
        : graph.max_out_degree();
    if (dmax <= 0) {
        return 0.1;
    }
    return 1.0 / static_cast<double>(dmax);
}

std::vector<RankedNode> rank_scores(const std::vector<double> &scores) {
    std::vector<RankedNode> ranking(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        ranking[i] = RankedNode{i, scores[i]};
    }
    std::sort(ranking.begin(), ranking.end(), ranked_node_better);
    return ranking;
}

std::vector<RankedNode> rank_top_scores(const std::vector<double> &scores, int limit) {
    const std::size_t capped_limit = std::min<std::size_t>(scores.size(), std::max(limit, 0));
    if (capped_limit == 0) {
        return {};
    }

    std::priority_queue<RankedNode, std::vector<RankedNode>, decltype(&ranked_node_better)> heap(
        ranked_node_better
    );
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const RankedNode candidate{i, scores[i]};
        if (heap.size() < capped_limit) {
            heap.push(candidate);
            continue;
        }
        if (ranked_node_better(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }

    std::vector<RankedNode> ranking;
    ranking.reserve(heap.size());
    while (!heap.empty()) {
        ranking.push_back(heap.top());
        heap.pop();
    }
    std::sort(ranking.begin(), ranking.end(), ranked_node_better);
    return ranking;
}

std::vector<RankedNode> tie_aware_top_k_ranking(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
) {
    const std::size_t capped_k = std::min<std::size_t>(scores.size(), std::max(k, 0));
    if (capped_k == 0 || scores.empty()) {
        return {};
    }
    if (target_counts.empty()) {
        return rank_top_scores(scores, k);
    }

    const std::vector<RankedNode> ranking = rank_scores(scores);
    const double boundary_score = ranking[capped_k - 1].score;

    std::vector<RankedNode> selected;
    selected.reserve(capped_k);
    std::vector<RankedNode> boundary_tied;
    std::map<std::string, int> fixed_counts;
    for (const RankedNode &item : ranking) {
        if (item.score > boundary_score + kEps) {
            selected.push_back(item);
            fixed_counts[graph.groups[item.index]] += 1;
        } else if (std::abs(item.score - boundary_score) <= kEps) {
            boundary_tied.push_back(item);
        } else {
            break;
        }
    }

    const std::size_t slots = capped_k - selected.size();
    if (slots >= boundary_tied.size()) {
        selected.insert(selected.end(), boundary_tied.begin(), boundary_tied.end());
        std::sort(selected.begin(), selected.end(), ranked_node_better);
        return selected;
    }

    const std::optional<std::string> protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value()) {
        selected.insert(selected.end(), boundary_tied.begin(), boundary_tied.begin() + slots);
        std::sort(selected.begin(), selected.end(), ranked_node_better);
        return selected;
    }

    std::vector<RankedNode> protected_tied;
    std::vector<RankedNode> other_tied;
    protected_tied.reserve(boundary_tied.size());
    other_tied.reserve(boundary_tied.size());
    for (const RankedNode &item : boundary_tied) {
        if (graph.groups[item.index] == *protected_group) {
            protected_tied.push_back(item);
        } else {
            other_tied.push_back(item);
        }
    }

    const int min_protected = std::max<int>(
        0,
        static_cast<int>(slots) - static_cast<int>(other_tied.size())
    );
    const int max_protected = std::min<int>(
        static_cast<int>(slots),
        static_cast<int>(protected_tied.size())
    );
    std::vector<RankedNode> best_tied;
    double best_gap = std::numeric_limits<double>::infinity();
    bool has_best = false;
    const auto lexicographic_node_order = [](const std::vector<RankedNode> &lhs,
                                             const std::vector<RankedNode> &rhs) {
        return std::lexicographical_compare(
            lhs.begin(),
            lhs.end(),
            rhs.begin(),
            rhs.end(),
            [](const RankedNode &a, const RankedNode &b) {
                return a.index < b.index;
            }
        );
    };

    for (int take_protected = min_protected; take_protected <= max_protected; ++take_protected) {
        const int take_other = static_cast<int>(slots) - take_protected;
        std::vector<RankedNode> candidate_tied;
        candidate_tied.reserve(slots);
        candidate_tied.insert(
            candidate_tied.end(),
            protected_tied.begin(),
            protected_tied.begin() + take_protected
        );
        candidate_tied.insert(
            candidate_tied.end(),
            other_tied.begin(),
            other_tied.begin() + take_other
        );
        std::sort(candidate_tied.begin(), candidate_tied.end(), ranked_node_better);

        std::map<std::string, int> candidate_counts = fixed_counts;
        for (const RankedNode &item : candidate_tied) {
            candidate_counts[graph.groups[item.index]] += 1;
        }
        const double candidate_gap = fairness_gap(candidate_counts, target_counts);
        if (!has_best ||
            candidate_gap < best_gap - kEps ||
            (std::abs(candidate_gap - best_gap) <= kEps &&
             lexicographic_node_order(candidate_tied, best_tied))) {
            has_best = true;
            best_gap = candidate_gap;
            best_tied = std::move(candidate_tied);
        }
    }

    selected.insert(selected.end(), best_tied.begin(), best_tied.end());
    std::sort(selected.begin(), selected.end(), ranked_node_better);
    return selected;
}

std::vector<RankedNode> tie_aware_full_ranking(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
) {
    const std::vector<RankedNode> ranking = rank_scores(scores);
    const std::vector<RankedNode> top_k_ranking =
        tie_aware_top_k_ranking(graph, scores, k, target_counts);
    std::vector<char> selected(scores.size(), 0);
    std::vector<RankedNode> result;
    result.reserve(ranking.size());
    for (const RankedNode &item : top_k_ranking) {
        if (item.index < selected.size() && selected[item.index] == 0) {
            selected[item.index] = 1;
            result.push_back(item);
        }
    }
    for (const RankedNode &item : ranking) {
        if (item.index < selected.size() && selected[item.index] == 0) {
            result.push_back(item);
        }
    }
    return result;
}

std::vector<std::size_t> top_k_nodes(const std::vector<RankedNode> &ranking, int k) {
    std::vector<std::size_t> result;
    const std::size_t limit = std::min<std::size_t>(ranking.size(), std::max(k, 0));
    result.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        result.push_back(ranking[i].index);
    }
    return result;
}

std::vector<std::size_t> tie_aware_top_k_nodes(
    const Graph &graph,
    const std::vector<double> &scores,
    int k,
    const std::map<std::string, int> &target_counts
) {
    return top_k_nodes(tie_aware_top_k_ranking(graph, scores, k, target_counts), k);
}

std::map<std::string, int> group_counts(const Graph &graph, const std::vector<std::size_t> &nodes) {
    std::map<std::string, int> counts;
    for (std::size_t node : nodes) {
        counts[graph.groups[node]] += 1;
    }
    return counts;
}

std::optional<std::string> protected_group_name(const std::map<std::string, int> &target_counts) {
    if (target_counts.empty()) {
        return std::nullopt;
    }
    return std::min_element(target_counts.begin(), target_counts.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    })->first;
}

double protected_group_target_share(const std::map<std::string, int> &target_counts) {
    const auto protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value()) {
        return 0.0;
    }
    int total = 0;
    for (const auto &[_, count] : target_counts) {
        total += count;
    }
    if (total <= 0) {
        return 0.0;
    }
    const auto it = target_counts.find(*protected_group);
    const int protected_count = it == target_counts.end() ? 0 : it->second;
    return static_cast<double>(protected_count) / static_cast<double>(total);
}

double fairness_gap(
    const std::map<std::string, int> &counts,
    const std::map<std::string, int> &target_counts
) {
    const auto protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value()) {
        return 0.0;
    }
    int k = 0;
    for (const auto &[_, count] : target_counts) {
        k += count;
    }
    if (k <= 0) {
        return 0.0;
    }
    const int realized_count = counts.count(*protected_group) ? counts.at(*protected_group) : 0;
    const double p = static_cast<double>(realized_count) / static_cast<double>(k);
    const double pi = protected_group_target_share(target_counts);
    const double deviation = p - pi;
    return deviation * deviation;
}

double katz_mass_share(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
) {
    const auto protected_group = protected_group_name(target_counts);
    if (!protected_group.has_value() || scores.size() != graph.nodes.size()) {
        return 0.0;
    }

    double protected_mass = 0.0;
    double total_mass = 0.0;
    for (std::size_t node = 0; node < scores.size(); ++node) {
        total_mass += scores[node];
        if (graph.groups[node] == *protected_group) {
            protected_mass += scores[node];
        }
    }
    if (std::abs(total_mass) <= kEps) {
        return 0.0;
    }
    return protected_mass / total_mass;
}

double katz_mass_objective(
    const Graph &graph,
    const std::vector<double> &scores,
    const std::map<std::string, int> &target_counts
) {
    const double mass_share = katz_mass_share(graph, scores, target_counts);
    const double pi = protected_group_target_share(target_counts);
    const double deviation = mass_share - pi;
    return deviation * deviation;
}

RankingState build_ranking_state(
    const Graph &graph,
    const CentralityComputation &centrality,
    int k,
    const std::map<std::string, int> &target_counts
) {
    RankingState state;
    state.centrality = centrality;
    state.top_k_ranking = tie_aware_top_k_ranking(graph, centrality.scores, k, target_counts);
    state.ranking = tie_aware_full_ranking(graph, centrality.scores, k, target_counts);
    state.top_k = top_k_nodes(state.top_k_ranking, k);
    state.counts = group_counts(graph, state.top_k);
    state.gap = fairness_gap(state.counts, target_counts);
    return state;
}

RankingState compute_ranking_state(
    const Graph &graph,
    const ExperimentOptions &options,
    int k,
    const std::map<std::string, int> &target_counts,
    std::optional<double> alpha_override
) {
    return build_ranking_state(
        graph,
        compute_katz_centrality(graph, options, alpha_override),
        k,
        target_counts
    );
}

}  // namespace top_k

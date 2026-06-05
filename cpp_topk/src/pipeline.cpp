#include "src/types.h"

#include <chrono>
#include <iostream>
#include <utility>

namespace top_k {

ExperimentSetup prepare_experiment_setup(int argc, char **argv) {
    ExperimentSetup setup;
    setup.options = parse_args(argc, argv);
    setup.dataset = resolve_dataset(setup.options);
    setup.graph = load_graph(setup.dataset);
    return setup;
}

ExperimentReport run_single_experiment(
    Graph &graph,
    const DatasetSpec &dataset,
    const ExperimentOptions &options
) {
    const auto started = std::chrono::steady_clock::now();

    ExperimentReport report;
    report.dataset = dataset;
    report.options = options;
    report.initial_edge_count = graph.edge_count();
    RankingState initial = compute_ranking_state(graph, options, options.k, options.target_counts, options.alpha);

    report.initial.top_k = initial.top_k;
    report.initial.ranking = initial.ranking;
    report.initial.top_k_ranking = initial.top_k_ranking;
    report.initial.counts = initial.counts;
    report.initial.gap = initial.gap;
    report.initial.centrality.alpha = initial.centrality.alpha;
    report.initial.centrality.scores = initial.centrality.scores;
    report.initial.centrality.elapsed_seconds = initial.centrality.elapsed_seconds;
    report.initial.centrality.backend = initial.centrality.backend;

    report.fair_result = run_fair_algorithm(graph, std::move(initial), options);
    if (!report.fair_result.gap_trace.empty()) {
        const GapTraceEntry &last = report.fair_result.gap_trace.back();
        report.final.top_k = last.top_k;
        report.final.ranking = last.top_k_ranking;
        report.final.top_k_ranking = last.top_k_ranking;
        report.final.counts = last.counts;
        report.final.gap = last.gap;
        report.final.centrality.scores = report.fair_result.final_scores;
        if (report.final.centrality.scores.empty()) {
            report.final = compute_ranking_state(
                graph,
                options,
                options.k,
                options.target_counts,
                report.initial.centrality.alpha
            );
        }
    } else {
        report.final = compute_ranking_state(
            graph,
            options,
            options.k,
            options.target_counts,
            report.initial.centrality.alpha
        );
    }

    const auto finished = std::chrono::steady_clock::now();
    report.total_elapsed_seconds = std::chrono::duration<double>(finished - started).count();
    return report;
}

int run_cli(int argc, char **argv) {
    ExperimentSetup setup = prepare_experiment_setup(argc, argv);
    const ExperimentReport report =
        run_single_experiment(setup.graph, setup.dataset, setup.options);
    print_summary(setup.graph, report);
    std::cout << "\n";
    return 0;
}

}  // namespace top_k

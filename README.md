# Fair Top-k Katz Centrality via Graph Design

This release contains the C++ experiment code and the datasets used for fair
top-k Katz centrality graph-design experiments.

## Repository Layout

- `cpp_topk/`: C++ runners, algorithms, and batch scripts.
- `data/`: real datasets and the BPA synthetic graph generator.

Generated files are not part of the source release. C++ binaries are written to
`cpp_topk/build/`, and experiment logs and summaries are written to
`cpp_topk/results/`.

## Requirements

- Linux or macOS shell environment.
- `g++` with C++17 support.
- OpenMP support for `top_k_runner` (`libgomp` on many Linux systems).
- Python 3 for helper code inside batch scripts.
- Optional: NumPy for faster synthetic BPA data generation.

Eigen 3.4.0 is vendored under `cpp_topk/third_party_lib/eigen-3.4.0`, so no
separate Eigen install is required.

## Build

From the repository root:

```bash
cd cpp_topk
bash build.sh
```

This builds:

- `cpp_topk/build/top_k_runner`
- `cpp_topk/build/pagerank_baselines_runner`

If `-march=native` is not supported on your machine, override the flags:

```bash
CXXFLAGS="-std=c++17 -O3" bash build.sh
```

## Data

Each real dataset directory has:

- `edges.txt`: whitespace-separated directed edge list, one `source target`
  pair per line, no header.
- `groups.txt`: whitespace-separated node group labels, one `node group` pair
  per line, no header.

Data folder should include real datasets:

```text
Blogs
Hopkins
Retweet
Deezer
Penn
Pokec
```

**Please note: Due to GitHub’s restrictions, you will need to download Pokec from https://snap.stanford.edu/data/soc-Pokec.html and rename the file to match the names of the other files.**

For named datasets, `top_k_runner` reads from `../data/<Dataset>/edges.txt` and
`../data/<Dataset>/groups.txt` by default when run inside `cpp_topk/`.

## Single Runs

Example BLADE run:

```bash
cd cpp_topk
./build/top_k_runner \
  --dataset Blogs \
  --data-root ../data \
  --algorithm blade \
  --k 100 \
  --target 0:50,1:50 \
  --frontier-limit 2
```

Example samegroup run:

```bash
./build/top_k_runner \
  --dataset Blogs \
  --data-root ../data \
  --algorithm same_group_support \
  --k 100 \
  --target 0:50,1:50
```

Example PageRank baseline run:

```bash
./build/pagerank_baselines_runner \
  --edge-path ../data/Blogs/edges.txt \
  --group-path ../data/Blogs/groups.txt \
  --algorithm pagerank_fairwalk \
  --top-k 100 \
  --top-k-max 0 \
  --target0 0.5
```

Main `top_k_runner` algorithm names:

| README name | CLI value |
| --- | --- |
| dense | `--algorithm dense` |
| BLADE | `--algorithm blade` |
| BLADE no batch | `--algorithm blade_no_batch` |
| samegroup | `--algorithm same_group_support` |
| KatzMass | `--algorithm katz_mass` |
| gap greedy | `--algorithm gap_greedy` |
| optimal | `--algorithm optimal` |

Supported PageRank baseline algorithms:

```text
pagerank_fairgd
pagerank_fairwalk
pagerank_crosswalk
pagerank_lfprn
pagerank_lfpru
```

## Batch Runs

The real-dataset batch script runs the included datasets stage by stage:

```bash
cd cpp_topk
bash run_real_dataset.sh
```

The default order is BLADE with `q=2`, samegroup, KatzMass, PageRank baselines,
BLADE with `q=1`, then BLADE with `q=5`.

Useful overrides:

```bash
K=100 TIMEOUT_SECONDS=3600 bash run_real_dataset.sh
CASE_IDS="Blogs Hopkins Retweet" bash run_real_dataset.sh
OUT_DIR=results/test_real_dataset bash run_real_dataset.sh
EDGE_ADMISSIBILITY=two_hop bash run_real_dataset.sh
```

Common environment variables:

- `DATA_ROOT`: data directory, default `../data`.
- `TOP_K_RUNNER`: top-k executable, default `build/top_k_runner`.
- `PAGERANK_BASELINES_RUNNER`: PageRank baseline executable, default
  `build/pagerank_baselines_runner`.
- `OUT_DIR`: log directory, default `results/real_dataset`.
- `SUMMARY_FILE`: summary TSV path, default
  `results/real_dataset/run_real_dataset_summary.tsv`.
- `K`: requested top-k size, default `100`.
- `CASE_IDS`: whitespace-separated dataset subset.
- `EDGE_ADMISSIBILITY`: addable edge filter for graph-design algorithms,
  `any` or `two_hop`; default `any`.
- `TIMEOUT_SECONDS`: per-command timeout.

Use `bash run_real_dataset.sh --help` for the script's short usage message.

## Synthetic BPA Data

We generate directed BPA graphs with homophily $\rho=0.5$, attachment
parameter $m=2$, minority fraction $0.35$, target $\pi=0.5$, $k=6$, and
$\alpha=1/d_{\max}$.

Generate the BPA graphs for `n` in `{50, 100, 200}`:

```bash
for n in 50 100 200; do
  python3 data/generate_bpa_model.py --n "$n" --out-dir data/bpa
done
```

The generator writes:

- `bpa_<n>.txt`: edge list.
- `bpa_<n>_gender.txt`: node group labels.

Run one generated BPA graph with the experiment setting:

```bash
cd cpp_topk
./build/top_k_runner \
  --dataset bpa \
  --data-root ../data \
  --bpa-size 50 \
  --bpa-rho 0.5 \
  --algorithm blade \
  --k 6 \
  --target male:3,female:3 \
  --budget -1
```

## Custom Datasets

Use explicit paths with `--dataset custom`:

```bash
cd cpp_topk
./build/top_k_runner \
  --dataset custom \
  --edge-path /path/to/edges.txt \
  --group-path /path/to/groups.txt \
  --directed \
  --algorithm blade \
  --k 50 \
  --edge-admissibility any \
  --target male:25,female:25
```

## Third-Party Code

Eigen 3.4.0 is included in `cpp_topk/third_party_lib/eigen-3.4.0`. Its license
files are kept with the vendored source.

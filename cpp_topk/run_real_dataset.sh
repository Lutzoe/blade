#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../data}"
TOP_K_RUNNER="${TOP_K_RUNNER:-$ROOT_DIR/build/top_k_runner}"
PAGERANK_BASELINES_RUNNER="${PAGERANK_BASELINES_RUNNER:-$ROOT_DIR/build/pagerank_baselines_runner}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/real_dataset}"
SUMMARY_FILE="${SUMMARY_FILE:-$OUT_DIR/run_real_dataset_summary.tsv}"

K="${K:-100}"
BUDGET="${BUDGET:--1}"
BACKEND="${BACKEND:-eigen_direct}"
UPDATE_MODE="${UPDATE_MODE:-jacobi}"
EDGE_ADMISSIBILITY="${EDGE_ADMISSIBILITY:-any}"
SAME_GROUP_WINDOW="${SAME_GROUP_WINDOW:-100}"
KATZ_MASS_ATTEMPT_FRACTION="${KATZ_MASS_ATTEMPT_FRACTION:-0.001}"
KATZ_MASS_MAX_COMMITS="${KATZ_MASS_MAX_COMMITS:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
RUN_PAGERANK_LARGE_DATASETS="${RUN_PAGERANK_LARGE_DATASETS:-0}"

DEFAULT_CASE_IDS=(
  Blogs
  Hopkins
  Retweet
  Deezer
  Penn
  Pokec
)
if [[ -n "${CASE_IDS:-}" ]]; then
  read -r -a CASE_IDS_SELECTED <<<"$CASE_IDS"
else
  CASE_IDS_SELECTED=("${DEFAULT_CASE_IDS[@]}")
fi

TOPK_METHODS=(
  blade_q2
  same_group_support
  katz_mass
  blade_q1
  blade_q5
)
PAGERANK_ALGORITHMS=(
  pagerank_fairgd
  pagerank_fairwalk
  pagerank_crosswalk
  pagerank_lfprn
  pagerank_lfpru
)
RUN_COUNT=0

format_command() {
  printf '%q ' "$@"
}

usage() {
  cat <<EOF
Usage: bash run_real_dataset.sh [extra top_k_runner args]

Runs real datasets from DATA_ROOT with K=${K} by default. The default order is:
  BLADE q=2, samegroup, KatzMass, PageRank baselines, BLADE q=1, BLADE q=5

Common environment overrides:
  DATA_ROOT=/path/to/data
  OUT_DIR=/path/to/results
  K=100
  CASE_IDS="Blogs Hopkins Retweet ..."
  EDGE_ADMISSIBILITY=any
  TIMEOUT_SECONDS=3600
  DRY_RUN=1
  RUN_PAGERANK_LARGE_DATASETS=1

Expected data layout:
  DATA_ROOT/Blogs/edges.txt       DATA_ROOT/Blogs/groups.txt
  DATA_ROOT/Hopkins/edges.txt     DATA_ROOT/Hopkins/groups.txt
  ...
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

EXTRA_TOPK_ARGS=("$@")

require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "$label not found or not executable: $path" >&2
    exit 1
  fi
}

require_data_root() {
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data root not found: $DATA_ROOT" >&2
    exit 1
  fi
}

is_case_id() {
  case "$1" in
    Blogs|Hopkins|Retweet|Deezer|Penn|Twitch|DBLP|Pokec) return 0 ;;
    *) return 1 ;;
  esac
}

require_case_id() {
  if ! is_case_id "$1"; then
    echo "Unknown case: $1" >&2
    echo "Known cases: ${DEFAULT_CASE_IDS[*]}" >&2
    exit 2
  fi
}

case_dataset() {
  require_case_id "$1"
  echo "$1"
}

case_data_dir() {
  require_case_id "$1"
  echo "$1"
}

case_edge_file() {
  require_case_id "$1"
  echo "$DATA_ROOT/$(case_data_dir "$1")/edges.txt"
}

case_group_file() {
  require_case_id "$1"
  echo "$DATA_ROOT/$(case_data_dir "$1")/groups.txt"
}

case_nodes() {
  local group_file
  group_file="$(case_group_file "$1")"
  wc -l <"$group_file"
}

case_top_k_dataset_args() {
  require_case_id "$1"
  printf '%s\n' --dataset "$(case_dataset "$1")"
}

case_exists() {
  local case_id="$1"
  local edge_file
  local group_file
  edge_file="$(case_edge_file "$case_id")"
  group_file="$(case_group_file "$case_id")"
  [[ -f "$edge_file" && -f "$group_file" ]]
}

effective_k() {
  local nodes="$1"
  local k="$K"
  if (( k > nodes )); then
    k="$nodes"
  fi
  if (( k % 2 == 1 )); then
    k=$(( k - 1 ))
  fi
  if (( k < 2 )); then
    k=2
  fi
  echo "$k"
}

balanced_target() {
  local k="$1"
  local group_file="$2"
  python3 - "$k" "$group_file" <<'PY'
import sys

k = int(sys.argv[1])
group_file = sys.argv[2]
labels = {line.split()[1] for line in open(group_file) if line.split()}
if not labels:
    raise SystemExit("empty group labels")

def key(label: str) -> tuple[int, object]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)

ordered = sorted(labels, key=key)
base = k // len(ordered)
remainder = k % len(ordered)
counts = [base + (1 if idx < remainder else 0) for idx in range(len(ordered))]
print(",".join(f"{label}:{count}" for label, count in zip(ordered, counts)))
PY
}

pr_target0() {
  local k="$1"
  local group_file="$2"
  python3 - "$k" "$group_file" <<'PY'
import sys

k = int(sys.argv[1])
group_file = sys.argv[2]
labels = {line.split()[1] for line in open(group_file) if line.split()}
if len(labels) != 2:
    raise SystemExit(f"PR PageRank baselines expect 2 labels, found {len(labels)}: {sorted(labels)}")

def key(label: str) -> tuple[int, object]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)

ordered = sorted(labels, key=key)
base = k // 2
remainder = k % 2
first_count = base + (1 if remainder else 0)
print(f"{first_count / k:.12g}")
PY
}

top_k_method_algorithm() {
  case "$1" in
    blade_q*) echo "blade" ;;
    same_group_support) echo "same_group_support" ;;
    katz_mass) echo "katz_mass" ;;
    *)
      echo "Unknown top-k method: $1" >&2
      exit 2
      ;;
  esac
}

top_k_method_q() {
  case "$1" in
    blade_q1) echo "1" ;;
    blade_q2) echo "2" ;;
    blade_q5) echo "5" ;;
    *) echo "-" ;;
  esac
}

top_k_method_dir() {
  case "$1" in
    blade_q1) echo "blade_jacobi_q1" ;;
    blade_q2) echo "blade_jacobi_q2" ;;
    blade_q5) echo "blade_jacobi_q5" ;;
    same_group_support) echo "same_group_window_${SAME_GROUP_WINDOW}" ;;
    katz_mass) echo "katz_mass" ;;
    *)
      echo "Unknown top-k method: $1" >&2
      exit 2
      ;;
  esac
}

append_summary() {
  local status="$1"
  local dataset="$2"
  local case_id="$3"
  local nodes="$4"
  local k="$5"
  local method="$6"
  local algorithm="$7"
  local update_mode="$8"
  local q="$9"
  local exit_code="${10}"
  local log_file="${11}"
  local note="${12}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$dataset" "$case_id" "$nodes" "$k" "$method" "$algorithm" "$update_mode" "$q" "$exit_code" "$log_file" "$note" \
    >>"$SUMMARY_FILE"
}

skip_case() {
  local dataset="$1"
  local case_id="$2"
  local nodes="$3"
  local k="$4"
  local method="$5"
  local algorithm="$6"
  local q="$7"
  local note="$8"
  echo "Skipping case=$case_id method=$method: $note"
  append_summary "skipped" "$dataset" "$case_id" "$nodes" "$k" "$method" "$algorithm" "default" "$q" "-" "-" "$note"
}

run_logged_command() {
  local log_file="$1"
  shift
  local -a runner_cmd=("$@")
  local -a cmd=()
  if [[ -n "$TIMEOUT_SECONDS" ]]; then
    cmd=(
      timeout
      --signal=TERM
      --kill-after=10s
      "${TIMEOUT_SECONDS}s"
      "${runner_cmd[@]}"
    )
  else
    cmd=("${runner_cmd[@]}")
  fi

  printf 'command: ' >>"$log_file"
  format_command "${cmd[@]}" >>"$log_file"
  {
    echo
    echo
  } >>"$log_file"

  local exit_code
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry_run: command not executed" >>"$log_file"
    exit_code=0
  else
    set +e
    "${cmd[@]}" >>"$log_file" 2>&1
    exit_code=$?
    set -e
  fi

  {
    echo
    echo "exit_code: $exit_code"
    echo "finished_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >>"$log_file"
  return "$exit_code"
}

status_for_exit_code() {
  local exit_code="$1"
  case "$exit_code" in
    0) echo "ok" ;;
    124|137|143) echo "timeout" ;;
    *) echo "failed" ;;
  esac
}

note_for_exit_code() {
  local exit_code="$1"
  case "$exit_code" in
    0) echo "" ;;
    124|137|143) echo "hit_${TIMEOUT_SECONDS:-process}s_timeout_or_cancelled" ;;
    *) echo "runner_exit_${exit_code}" ;;
  esac
}

pagerank_baseline_skip_reason() {
  local case_id="$1"
  local nodes="$2"
  local algorithm="$3"
  if [[ "$RUN_PAGERANK_LARGE_DATASETS" == "1" ]]; then
    return
  fi
  case "$algorithm" in
    pagerank_fairgd)
      if (( nodes >= 40000 )); then
        echo "iterative_pagerank_gradient_too_large_for_default_batch"
      fi
      ;;
    pagerank_lfprn|pagerank_lfpru)
      if (( nodes >= 5000 )); then
        echo "dense_support_expansion_too_large_for_default_batch"
      fi
      ;;
    pagerank_crosswalk)
      if (( nodes >= 1000000 )); then
        echo "random_walk_preprocessing_too_large_for_default_batch"
      fi
      ;;
  esac
}

run_top_k_method() {
  local method="$1"
  local case_id="$2"
  local dataset="$3"
  local nodes="$4"
  local k="$5"
  local target="$6"
  local q
  q="$(top_k_method_q "$method")"
  local algorithm
  algorithm="$(top_k_method_algorithm "$method")"
  local method_dir
  method_dir="$(top_k_method_dir "$method")"
  local log_dir="$OUT_DIR/k_${k}/$method_dir"
  local log_file="$log_dir/${case_id}_${method_dir}.log"
  mkdir -p "$log_dir"

  mapfile -t dataset_args < <(case_top_k_dataset_args "$case_id")
  local -a runner_cmd=(
    "$TOP_K_RUNNER"
    "${EXTRA_TOPK_ARGS[@]}"
    "${dataset_args[@]}"
    --data-root "$DATA_ROOT"
    --backend "$BACKEND"
    --algorithm "$algorithm"
    --edge-admissibility "$EDGE_ADMISSIBILITY"
    --k "$k"
    --budget "$BUDGET"
    --target "$target"
  )
  if [[ "$algorithm" == "blade" ]]; then
    runner_cmd+=(--frontier-limit "$q" --update-mode "$UPDATE_MODE")
  elif [[ "$algorithm" == "same_group_support" ]]; then
    runner_cmd+=(--same-group-window "$SAME_GROUP_WINDOW" --update-mode "$UPDATE_MODE")
  elif [[ "$algorithm" == "katz_mass" ]]; then
    runner_cmd+=(--update-mode "$UPDATE_MODE")
    if [[ -n "$KATZ_MASS_ATTEMPT_FRACTION" && "$KATZ_MASS_ATTEMPT_FRACTION" != "0" ]]; then
      runner_cmd+=(--katz-mass-attempt-fraction "$KATZ_MASS_ATTEMPT_FRACTION")
    fi
    if [[ -n "$KATZ_MASS_MAX_COMMITS" ]]; then
      runner_cmd+=(--katz-mass-max-commits "$KATZ_MASS_MAX_COMMITS")
    fi
  fi

  {
    echo "start_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "dataset: $dataset"
    echo "case_id: $case_id"
    echo "nodes: $nodes"
    echo "k: $k"
    echo "target: $target"
    echo "method: $method"
    echo "algorithm: $algorithm"
    echo "edge_admissibility: $EDGE_ADMISSIBILITY"
    echo "update_mode: $UPDATE_MODE"
    if [[ "$q" != "-" ]]; then
      echo "q: $q"
    fi
    if [[ "$algorithm" == "same_group_support" ]]; then
      echo "same_group_window: $SAME_GROUP_WINDOW"
    fi
    if [[ "$algorithm" == "katz_mass" ]]; then
      echo "katz_mass_attempt_fraction: ${KATZ_MASS_ATTEMPT_FRACTION:-0}"
      echo "katz_mass_max_commits: ${KATZ_MASS_MAX_COMMITS:-unbounded}"
    fi
  } >"$log_file"

  echo "Running case=$case_id method=$method k=$k"
  local exit_code
  set +e
  run_logged_command "$log_file" "${runner_cmd[@]}"
  exit_code=$?
  set -e

  local status
  local note
  status="$(status_for_exit_code "$exit_code")"
  note="$(note_for_exit_code "$exit_code")"
  append_summary "$status" "$dataset" "$case_id" "$nodes" "$k" "$method" "$algorithm" "$UPDATE_MODE" "$q" "$exit_code" "$log_file" "$note"
  RUN_COUNT=$((RUN_COUNT + 1))
}

run_pagerank_baseline() {
  local algorithm="$1"
  local case_id="$2"
  local dataset="$3"
  local nodes="$4"
  local k="$5"
  local target0="$6"
  local edge_file
  local group_file
  edge_file="$(case_edge_file "$case_id")"
  group_file="$(case_group_file "$case_id")"
  local log_dir="$OUT_DIR/k_${k}/pagerank_baselines/$algorithm"
  local log_file="$log_dir/${case_id}_${algorithm}.log"
  mkdir -p "$log_dir"

  local skip_reason
  skip_reason="$(pagerank_baseline_skip_reason "$case_id" "$nodes" "$algorithm")"
  if [[ -n "$skip_reason" ]]; then
    skip_case "$dataset" "$case_id" "$nodes" "$k" "$algorithm" "$algorithm" "-" "$skip_reason"
    return
  fi

  local -a runner_cmd=(
    "$PAGERANK_BASELINES_RUNNER"
    --edge-path "$edge_file"
    --group-path "$group_file"
    --algorithm "$algorithm"
    --top-k "$k"
    --top-k-max 0
    --target0 "$target0"
  )

  {
    echo "start_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "dataset: $dataset"
    echo "case_id: $case_id"
    echo "nodes: $nodes"
    echo "k: $k"
    echo "target0: $target0"
    echo "method: $algorithm"
    echo "algorithm: $algorithm"
    echo "update_mode: pagerank_baselines"
  } >"$log_file"

  echo "Running case=$case_id method=$algorithm k=$k"
  local exit_code
  set +e
  run_logged_command "$log_file" "${runner_cmd[@]}"
  exit_code=$?
  set -e

  local status
  local note
  status="$(status_for_exit_code "$exit_code")"
  note="$(note_for_exit_code "$exit_code")"
  append_summary "$status" "$dataset" "$case_id" "$nodes" "$k" "$algorithm" "$algorithm" "pagerank_baselines" "-" "$exit_code" "$log_file" "$note"
  RUN_COUNT=$((RUN_COUNT + 1))
}

validate_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$name must be a positive integer, got: $value" >&2
    exit 1
  fi
}

validate_binary_flag() {
  local name="$1"
  local value="$2"
  case "$value" in
    0|1) ;;
    *)
      echo "$name must be 0 or 1, got: $value" >&2
      exit 1
      ;;
  esac
}

validate_integer K "$K"
validate_integer SAME_GROUP_WINDOW "$SAME_GROUP_WINDOW"
validate_binary_flag DRY_RUN "$DRY_RUN"
validate_binary_flag SKIP_MISSING "$SKIP_MISSING"
validate_binary_flag RUN_PAGERANK_LARGE_DATASETS "$RUN_PAGERANK_LARGE_DATASETS"
require_data_root
require_executable "$TOP_K_RUNNER" "Top-k runner"
require_executable "$PAGERANK_BASELINES_RUNNER" "PageRank baselines runner"

mkdir -p "$OUT_DIR" "$(dirname "$SUMMARY_FILE")"
printf 'status\tdataset\tcase_id\tnodes\tk\tmethod\talgorithm\tupdate_mode\tq\texit_code\tlog_file\tnote\n' >"$SUMMARY_FILE"

echo "Data root: $DATA_ROOT"
echo "Output dir: $OUT_DIR"
echo "Summary: $SUMMARY_FILE"
echo "K: $K"
echo "Cases: ${CASE_IDS_SELECTED[*]}"
echo "Edge admissibility: $EDGE_ADMISSIBILITY"
echo "Method order: all cases BLADE q=2, then samegroup, KatzMass, pagerank_baselines, BLADE q=1, BLADE q=5"
if [[ -n "$TIMEOUT_SECONDS" ]]; then
  echo "Timeout per run: ${TIMEOUT_SECONDS}s"
else
  echo "Timeout per run: none"
fi
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: commands will be logged but not executed"
fi

AVAILABLE_CASE_IDS=()
for case_id in "${CASE_IDS_SELECTED[@]}"; do
  dataset="$(case_dataset "$case_id")"
  if ! case_exists "$case_id"; then
    if [[ "$SKIP_MISSING" == "1" ]]; then
      echo "Skipping case=$case_id because edge or group file is missing."
      append_summary "skipped" "$dataset" "$case_id" "-" "-" "all" "all" "default" "-" "-" "-" "missing_edge_or_group_file"
      continue
    fi
    echo "Missing edge or group file for case=$case_id" >&2
    exit 1
  fi
  AVAILABLE_CASE_IDS+=("$case_id")
done

run_top_k_stage() {
  local method="$1"
  local case_id
  for case_id in "${AVAILABLE_CASE_IDS[@]}"; do
    local dataset
    local nodes
    local k
    local group_file
    local target
    dataset="$(case_dataset "$case_id")"
    nodes="$(case_nodes "$case_id")"
    nodes="${nodes//[[:space:]]/}"
    k="$(effective_k "$nodes")"
    group_file="$(case_group_file "$case_id")"
    target="$(balanced_target "$k" "$group_file")"
    run_top_k_method "$method" "$case_id" "$dataset" "$nodes" "$k" "$target"
  done
}

run_pagerank_stage() {
  local pagerank_algorithm="$1"
  local case_id
  for case_id in "${AVAILABLE_CASE_IDS[@]}"; do
    local dataset
    local nodes
    local k
    local group_file
    local target0
    dataset="$(case_dataset "$case_id")"
    nodes="$(case_nodes "$case_id")"
    nodes="${nodes//[[:space:]]/}"
    k="$(effective_k "$nodes")"
    group_file="$(case_group_file "$case_id")"
    target0="$(pr_target0 "$k" "$group_file")"
    run_pagerank_baseline "$pagerank_algorithm" "$case_id" "$dataset" "$nodes" "$k" "$target0"
  done
}

run_top_k_stage blade_q2
run_top_k_stage same_group_support
run_top_k_stage katz_mass
for pagerank_algorithm in "${PAGERANK_ALGORITHMS[@]}"; do
  run_pagerank_stage "$pagerank_algorithm"
done
run_top_k_stage blade_q1
run_top_k_stage blade_q5

echo "Finished. Executed $RUN_COUNT experiments."
echo "Summary written to $SUMMARY_FILE"

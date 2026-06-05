#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O3 -march=native}"
OPENMP_FLAGS="${OPENMP_FLAGS:--fopenmp}"
EIGEN_DIR="${EIGEN_DIR:-$ROOT_DIR/third_party_lib/eigen-3.4.0}"

mkdir -p "$BUILD_DIR"

echo "Building top_k_runner..."
"$CXX" $CXXFLAGS $OPENMP_FLAGS \
  -I"$ROOT_DIR" -I"$EIGEN_DIR" \
  "$ROOT_DIR/src/main.cpp" \
  "$ROOT_DIR/src/cli.cpp" \
  "$ROOT_DIR/src/graph.cpp" \
  "$ROOT_DIR/src/pipeline.cpp" \
  "$ROOT_DIR/src/dispatch.cpp" \
  "$ROOT_DIR/katz_calculation/katz.cpp" \
  "$ROOT_DIR/katz_calculation/eigen/eigen.cpp" \
  "$ROOT_DIR/algorithms/helpers/helpers.cpp" \
  "$ROOT_DIR/algorithms/helpers/updates.cpp" \
  "$ROOT_DIR/algorithms/gap_greedy/gap_greedy.cpp" \
  "$ROOT_DIR/algorithms/katz_mass/katz_mass.cpp" \
  "$ROOT_DIR/algorithms/optimal/optimal.cpp" \
  "$ROOT_DIR/algorithms/dense/dense.cpp" \
  "$ROOT_DIR/algorithms/blade/blade.cpp" \
  "$ROOT_DIR/algorithms/same_group_support/same_group_support.cpp" \
  -o "$BUILD_DIR/top_k_runner"

echo "Building pagerank_baselines_runner..."
"$CXX" $CXXFLAGS \
  -I"$ROOT_DIR" \
  "$ROOT_DIR/algorithms/pagerank_baselines/pagerank_baselines.cpp" \
  -o "$BUILD_DIR/pagerank_baselines_runner"

echo "Built:"
echo "  $BUILD_DIR/top_k_runner"
echo "  $BUILD_DIR/pagerank_baselines_runner"

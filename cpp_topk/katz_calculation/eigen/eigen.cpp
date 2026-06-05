#include "katz_calculation/katz.h"

#include <algorithm>

#include <Eigen/Dense>

namespace top_k {
namespace {

Eigen::MatrixXd adjacency_matrix(const Graph &graph) {
    const Eigen::Index n = static_cast<Eigen::Index>(graph.nodes.size());
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n, n);
    for (std::size_t u = 0; u < graph.nodes.size(); ++u) {
        for (std::size_t v : graph.out_neighbors[u]) {
            matrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) = 1.0;
        }
    }
    return matrix;
}

DenseMatrix to_dense_matrix(const Eigen::MatrixXd &matrix) {
    const std::size_t n = static_cast<std::size_t>(matrix.rows());
    DenseMatrix dense(n);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            dense.at(static_cast<std::size_t>(row), static_cast<std::size_t>(col)) = matrix(row, col);
        }
    }
    return dense;
}

}  // namespace

DenseMatrix compute_eigen_direct_katz_matrix(const Graph &graph, double alpha) {
    const Eigen::MatrixXd adjacency = adjacency_matrix(graph);
    const Eigen::Index n = adjacency.rows();
    Eigen::MatrixXd system = Eigen::MatrixXd::Identity(n, n) - alpha * adjacency;

    Eigen::MatrixXd katz = system.inverse();
    for (Eigen::Index row = 0; row < n; ++row) {
        for (Eigen::Index col = 0; col < n; ++col) {
            katz(row, col) = std::max(katz(row, col), kEps);
        }
    }
    return to_dense_matrix(katz);
}

}  // namespace top_k

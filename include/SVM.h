#pragma once

#include <cstddef>
#include <vector>
namespace SVM {
constexpr double DEFAULT_C = 1.0;
constexpr double DEFAULT_SIGMA = 1.0;
constexpr std::size_t DEFAULT_MAX_ITER = 100;
constexpr double DEFAULT_TOL = 1e-3;
}  // namespace SVM

/**
 * @brief gaussian_kernel
 * @param x The first matrix
 * @param y The second matrix
 * @param sigma Calculation parameter,the default value is 1.0
 * @return The calculation result, if calculation is error, return -1.0
 */
auto gaussian_kernel(const std::vector<double> &x, const std::vector<double> &y,
                     double sigma = SVM::DEFAULT_SIGMA) noexcept -> double;

auto dual_svm(const std::vector<std::vector<double>> &x,
              const std::vector<double> &y, double C = SVM::DEFAULT_C,
              double sigma = SVM::DEFAULT_SIGMA,
              std::size_t max_iter = SVM::DEFAULT_MAX_ITER,
              double tol = SVM::DEFAULT_TOL) noexcept
    -> std::pair<std::vector<double>, double>;
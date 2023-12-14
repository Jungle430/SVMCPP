#include "SVM.h"

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

auto gaussian_kernel(const std::vector<double> &x, const std::vector<double> &y,
                     double sigma) noexcept -> double {
  double norm = 0.0;
  if (x.size() != y.size()) {
    return -1.0;
  }
  auto n = x.size();
  for (auto i = 0; i < n; i++) {
    norm += std::pow(x[i] - y[i], 2);
  }
  return std::exp(-norm / (2 * std::pow(sigma, 2)));
}

auto dual_svm(const std::vector<std::vector<double>> &x,
              const std::vector<double> &y, double C, double sigma,
              std::size_t max_iter, double tol) noexcept
    -> std::pair<std::vector<double>, double> {
  const auto m = x.size();
  const auto n = x[0].size();

  auto alpha = std::vector<double>(m, 0.0);
  double b = 0.0;

  auto kernel_matrix =
      std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));

  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < m; j++) {
      kernel_matrix[i][j] = gaussian_kernel(x[i], x[j], sigma);
    }
  }

  for (auto iter = 0; iter < max_iter; iter++) {
    for (auto i = 0; i < m; i++) {
      double error_i = 0.0;
      for (auto k = 0; k < m; k++) {
        error_i += alpha[k] * y[k] * kernel_matrix[i][k];
      }
      error_i += b - y[i];

      if ((y[i] * error_i < -tol && alpha[i] < C) ||
          (y[i] * error_i > tol && alpha[i] > 0)) {
        size_t j = std::rand() % m;
        double error_j = 0.0;
        for (auto k = 0; k < m; k++) {
          error_j += alpha[k] * y[k] * kernel_matrix[j][k];
        }
        error_j += b - y[j];

        double alpha_i_old = alpha[i];
        double alpha_j_old = alpha[j];

        double L = 0.0;
        double H = 0.0;
        if (y[i] != y[j]) {
          L = std::max(0.0, alpha[j] - alpha[i]);
          H = std::min(C, C + alpha[j] - alpha[i]);
        } else {
          L = std::max(0.0, alpha[i] + alpha[j] - C);
          H = std::min(C, alpha[i] + alpha[j]);
        }

        double eta =
            2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j];

        if (eta >= 0) {
          continue;
        }

        alpha[j] = alpha[j] - (y[j] * (error_i - error_j)) / eta;
        alpha[j] = std::max(L, std::min(H, alpha[j]));

        alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j]);

        double b1 = b - error_i -
                    y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][i] -
                    y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[i][j];
        double b2 = b - error_j -
                    y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][j] -
                    y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[j][j];

        if (0 < alpha[i] && alpha[i] < C) {
          b = b1;
        } else if (0 < alpha[j] && alpha[j] < C) {
          b = b2;
        } else {
          b = (b1 + b2) / 2;
        }
      }
    }
  }
  return {alpha, b};
}
#pragma once

#include <cmath>
#include <cstddef>
#include <set>
#include <unordered_map>
#include <vector>

namespace SVM {
constexpr double DEFAULT_C = 1.0;
constexpr std::size_t DEFAULT_MAX_ITER = 200;
constexpr double DEFAULT_TOL = 1e-3;
}  // namespace SVM

/**
 * @brief SVM, using SMO algorithm
 * @param y Labels for each sample(+1 or -1)
 * @param kernel_function Kernel function
 * @param C Regularization parameter
 * @param max_iter maximum iterations
 * @param tol Tolerance, which controls the stop condition of the iteration
 */
template <typename KERNEL_FUNCTION>
auto dual_svm(const std::vector<std::vector<double>> &x,
              const std::vector<int> &y, const KERNEL_FUNCTION &kernel_function,
              double C = SVM::DEFAULT_C,
              std::size_t max_iter = SVM::DEFAULT_MAX_ITER,
              double tol = SVM::DEFAULT_TOL) noexcept
    -> std::pair<std::vector<double>, double> {
  // value
  const auto m = x.size();
  // label
  const auto n = x[0].size();

  // vector alpha
  auto alpha = std::vector<double>(m, 0.0);

  // Threshold value b
  double b = 0.0;

  // build the kernel matrix
  auto kernel_matrix =
      std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));

  // using kernel function
  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < m; j++) {
      kernel_matrix[i][j] = kernel_function(x[i], x[j]);
    }
  }
  // build the kernel finished

  // main loop of SMO algorithm
  for (auto iter = 0; iter < max_iter; iter++) {
    for (auto i = 0; i < m; i++) {
      double error_i = 0.0;
      for (auto k = 0; k < m; k++) {
        error_i += alpha[k] * y[k] * kernel_matrix[i][k];
      }
      error_i += b - y[i];

      if ((y[i] * error_i < -tol && alpha[i] < C) ||
          (y[i] * error_i > tol && alpha[i] > 0)) {
        size_t j = rand() % m;
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

/**
 * @brief test SVM accuracy
 * @param x test data
 * @param y test label(+1,-1)
 * @param alpha SVM alpha
 * @param b SVM b
 * @param kernel_function Kernel function
 */
template <typename KERNEL_FUNCTION>
auto test_svm(const std::vector<std::vector<double>> &x,
              const std::vector<int> &y, const std::vector<double> &alpha,
              double b, const KERNEL_FUNCTION &kernel_function) noexcept
    -> double {
  const auto m = x.size();
  double correct_predictions = 0.0;

  for (auto i = 0; i < m; i++) {
    double prediction = b;

    for (auto j = 0; j < m; j++) {
      prediction += alpha[j] * y[j] * kernel_function(x[j], x[i]);
    }

    // Classify as +1 if prediction is greater than 0, else classify as -1
    int predicted_label = (prediction > 0) ? 1 : -1;

    // Check if the predicted label matches the true label
    if (predicted_label == y[i]) {
      correct_predictions += 1.0;
    }
  }

  // Calculate accuracy as the ratio of correct predictions to total predictions
  double accuracy = correct_predictions / static_cast<double>(m);
  return accuracy;
}

template <typename KERNEL_FUNCTION>
auto SVM_prediction(const std::vector<std::vector<double>> &x_train,
                    const std::vector<int> &y_train,
                    const std::vector<double> &alpha, const double b,
                    const KERNEL_FUNCTION &kernel_function,
                    const std::vector<double> new_data) noexcept -> double {
  const auto m = x_train.size();
  double predicted = b;
  for (auto j = 0; j < m; j++) {
    predicted += alpha[j] * y_train[j] * kernel_function(x_train[j], new_data);
  }
  return predicted;
}

template <typename KERNEL_FUNCTION>
auto SVM_prediction_number(const std::vector<std::vector<double>> &x_train,
                           const std::vector<int> &y_train,
                           const std::vector<double> &alpha, const double b,
                           const KERNEL_FUNCTION &kernel_function,
                           const std::vector<double> new_data,
                           int number) noexcept -> double {
  auto new_y_train = std::vector<int>(y_train.size(), 0.0);
  const auto n = y_train.size();
  for (auto i = 0; i < n; i++) {
    new_y_train[i] = y_train[i] == number ? 1 : -1;
  }
  return SVM_prediction(x_train, new_y_train, alpha, b, kernel_function,
                        new_data);
}
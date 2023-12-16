#pragma once

#include <vector>

namespace SVM {

namespace GAUSSIAN_KERNEL {
constexpr double DEFAULT_SIGMA = 1.0;
};

class gaussian_kernel {
 private:
  double sigma = SVM::GAUSSIAN_KERNEL::DEFAULT_SIGMA;

 public:
  gaussian_kernel() = default;

  explicit gaussian_kernel(double sigma) : sigma(sigma){};

  [[nodiscard]] auto getSigma() const -> double;

  auto setSigma(double sigma) -> void;
  /**
   * @brief gaussian_kernel
   * @param x The first matrix
   * @param y The second matrix
   * @param sigma Calculation parameter,the default value is 1.0
   * @return The calculation result, if calculation is error, return -1.0
   */
  auto operator()(const std::vector<double> &x,
                  const std::vector<double> &y) const noexcept -> double;
};
}  // namespace SVM
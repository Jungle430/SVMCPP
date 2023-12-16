#pragma once

#include <vector>

namespace SVM {
namespace RBF_KERNEL {
constexpr double DEFAULT_GAMMA = 1.0;
};

class rbf_kernel {
 private:
  double gamma = SVM::RBF_KERNEL::DEFAULT_GAMMA;

 public:
  rbf_kernel() = default;

  explicit rbf_kernel(double gamma) : gamma(gamma) {}

  [[nodiscard]] auto getSigma() const -> double;

  auto setSigma(double gamma) -> void;

  /**
   * @brief rbf_kernel
   * @param x The first matrix
   * @param y The second matrix
   * @param sigma Calculation parameter,the default value is 1.0
   * @return The calculation result, if calculation is error, return -1.0
   */
  auto operator()(const std::vector<double> &x,
                  const std::vector<double> &y) const noexcept -> double;
};

}  // namespace SVM
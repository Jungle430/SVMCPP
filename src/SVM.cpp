#include "SVM.h"

auto SVM::gaussian_kernel::operator()(const std::vector<double> &x,
                                      const std::vector<double> &y,
                                      double sigma) const noexcept -> double {
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
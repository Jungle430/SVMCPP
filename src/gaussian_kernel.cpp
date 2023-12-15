#include "gaussian_kernel.h"

#include <cmath>

auto SVM::gaussian_kernel::operator()(
    const std::vector<double> &x, const std::vector<double> &y) const noexcept
    -> double {
  double norm = 0.0;
  if (x.size() != y.size()) {
    return -1.0;
  }
  auto n = x.size();
  for (auto i = 0; i < n; i++) {
    norm += std::pow(x[i] - y[i], 2);
  }
  return std::exp(-norm / (2 * std::pow(this->sigma, 2)));
}

auto SVM::gaussian_kernel::getSigma() const -> double { return this->sigma; }

auto SVM::gaussian_kernel::setSigma(double sigma) -> void {
  this->sigma = sigma;
}
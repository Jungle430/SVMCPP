#include "rbf_kernel.h"

#include <cmath>

auto SVM::rbf_kernel::operator()(const std::vector<double> &x,
                                 const std::vector<double> &y) const noexcept
    -> double {
  if (x.size() != y.size()) {
    return -1.0;
  }

  double distance = 0.0;
  auto n = x.size();

  for (auto i = 0; i < n; i++) {
    double diff = x[i] - y[i];
    distance += diff * diff;
  }
  return std::exp(-this->gamma * distance);
}

auto SVM::rbf_kernel::setSigma(double gamma) -> void {
  this->gamma = gamma;
}

auto SVM::rbf_kernel::getSigma() const -> double {
  return this->gamma;
}
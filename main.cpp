#include <cstdlib>
#include <ctime>
#include <iostream>

#include "SVM.h"  // Include the header file where SVM functions are declared

auto main() -> int {
  // Set a seed for random number generation
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  // Generate random data
  const std::size_t data_size = 100;
  const std::size_t feature_size = 2;
  std::vector<std::vector<double>> data;
  std::vector<double> labels;

  for (std::size_t i = 0; i < data_size; ++i) {
    std::vector<double> point;
    point.push_back(static_cast<double>(std::rand()) / RAND_MAX);
    point.push_back(static_cast<double>(std::rand()) / RAND_MAX);
    data.push_back(point);

    // Assign labels randomly (1 or -1)
    labels.push_back((std::rand() % 2 == 0) ? 1.0 : -1.0);
  }

  // Define the Gaussian kernel
  SVM::gaussian_kernel kernel;

  // Call the dual_svm function to perform SVM calculations
  auto result = dual_svm(data, labels, kernel);

  // Extract the support vector weights and bias
  std::vector<double> weights = result.first;
  double bias = result.second;

  // Print the results
  std::cout << "Support Vector Weights: ";
  for (const auto &weight : weights) {
    std::cout << weight << " ";
  }
  std::cout << "\nBias: " << bias << std::endl;

  return 0;
}

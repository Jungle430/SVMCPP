#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "CSV.h"
#include "SVM.h"  // Include the header file where SVM functions are declared
#include "gaussian_kernel.h"
#include "rbf_kernel.h"

constexpr auto start = 0;
constexpr auto end = 9;

const auto train_images_path =
    std::string("../read_file_script/train_images.csv");
const auto train_labels_path =
    std::string("../read_file_script/train_labels.csv");
const auto test_images_path =
    std::string("../read_file_script/test_images.csv");
const auto test_labels_path =
    std::string("../read_file_script/test_labels.csv");
const auto outputFile = std::string("../model_data.csv");

auto train(int number) -> std::pair<std::vector<double>, double>;
auto test(const std::vector<double> &alpha, double b, int number) -> double;

auto main() -> int {
  auto results = std::vector<std::pair<std::vector<double>, double>>();
  for (auto i = start; i <= end; i++) {
    results.emplace_back(train(i));
  }
  CSV::writeCSV(outputFile, results);
  auto test_images = CSV::loadImages(test_images_path);
  auto test_labels = CSV::loadLabels(test_labels_path);
  auto rbfKernel = SVM::rbf_kernel();
  double count = 0.0;
  for (auto k = 0; k < test_images.size(); k++) {
    auto ans = std::vector<double>();
    for (auto i = start; i <= end; i++) {
      ans.emplace_back(SVM_prediction_number(
          test_images, test_labels, results[i].first, results[i].second,
          rbfKernel, test_images[k], i));
    }
    auto index = 0;
    auto min_ans = ans[0];
    for (auto h = 0; h < ans.size(); h++) {
      if (ans[h] > min_ans) {
        min_ans = ans[h];
        index = h;
      }
    }
    if (index == test_labels[k]) {
      count += 1;
    }
    std::cout << index << " " << test_labels[k] << std::endl;
  }
  std::cout << count << std::endl;
  std::cout << test_labels.size() << std::endl;
  std::cout << count / static_cast<double>(test_labels.size()) << std::endl;
  return 0;
}

auto train(int number) -> std::pair<std::vector<double>, double> {
  auto train_images = CSV::loadImages(train_images_path);
  auto train_labels = CSV::loadLabels(train_labels_path);
  for (auto &x : train_labels) {
    x = x == number ? 1 : -1;
  }
  auto rbfKernel = SVM::rbf_kernel();
  return dual_svm(train_images, train_labels, rbfKernel);
}

auto test(const std::vector<double> &alpha, double b, int number) -> double {
  auto test_images = CSV::loadImages(test_images_path);
  auto test_labels = CSV::loadLabels(test_labels_path);
  for (auto &x : test_labels) {
    x = x == number ? 1 : -1;
  }
  auto rbfKernel = SVM::rbf_kernel();
  return test_svm(test_images, test_labels, alpha, b, rbfKernel);
}
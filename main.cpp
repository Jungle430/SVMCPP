#include <cstdlib>
#include <iostream>
#include <utility>

#include "CSV.h"
#include "SVM.h"  // Include the header file where SVM functions are declared
#include "gaussian_kernel.h"
#include "rbf_kernel.h"

const auto train_images_path =
    std::string("../read_file_script/train_images.csv");
const auto train_labels_path =
    std::string("../read_file_script/train_labels.csv");
const auto test_images_path =
    std::string("../read_file_script/test_images.csv");
const auto test_labels_path =
    std::string("../read_file_script/test_labels.csv");

auto train(int number) -> std::pair<std::vector<double>, double>;
auto test(const std::vector<double> &alpha, double b, int number) -> double;

auto main(int argc, char **argv) -> int {
  if (argc < 2) {
    std::cout << "SVMCPP <Forecast figure(0~9)>" << std::endl;
  }

  int number = std::atoi(argv[1]);

  auto svmResult = train(number);
  // for (auto &x : svmResult.first) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;
  auto accuracy = test(svmResult.first, svmResult.second, number);
  std::cout << accuracy << std::endl;
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
#include <iostream>

#include "CSV.h"
#include "SVM.h"  // Include the header file where SVM functions are declared
#include "gaussian_kernel.h"

const auto train_images_path =
    std::string("../read_file_script/train_images.csv");
const auto train_labels_path =
    std::string("../read_file_script/train_labels.csv");
const auto test_images_path =
    std::string("../read_file_script/test_images.csv");
const auto test_labels_path =
    std::string("../read_file_script/test_labels.csv");

auto main() -> int {
  auto train_images = CSV::loadImages(train_images_path);
  auto train_labels = CSV::loadLabels(train_images_path);
  auto test_images = CSV::loadImages(test_images_path);
  auto test_labels = CSV::loadLabels(test_labels_path);
  auto gaussianKernel =
      SVM::gaussian_kernel(SVM::GAUSSIAN_KERNEL::DEFAULT_SIGMA);
  auto svmResult =
      multi_dimensional_svm(train_images, train_labels, gaussianKernel);
  for (auto x : svmResult.first) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  std::cout << svmResult.second << std::endl;
  return 0;
}

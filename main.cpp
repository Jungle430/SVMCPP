#include <iostream>

#include "SVM.h"

auto main() -> int {  // 生成示例数据
  std::vector<std::vector<double>> X = {{-1, -1}, {-2, -1}, {-3, -2},
                                        {1, 1},   {2, 1},   {3, 2}};
  std::vector<double> y = {-1, -1, -1, 1, 1, 1};

  // 设置参数
  double C = 1.0;
  double sigma = 1.0;

  // 调用支持向量机算法
  auto result = dual_svm(X, y, C, sigma);

  // 提取 alpha 和 b
  std::vector<double> alpha = result.first;
  double b = result.second;

  // 打印结果
  std::cout << "Alpha: ";
  for (const auto& a : alpha) {
    std::cout << a << " ";
  }
  std::cout << "\n";

  std::cout << "b: " << b << "\n";

  // 进行预测
  std::vector<std::vector<double>> new_samples = {{-2, -2}, {2, 2}};
  for (const auto& sample : new_samples) {
    double prediction = b;
    for (size_t i = 0; i < alpha.size(); ++i) {
      prediction += alpha[i] * y[i] * gaussian_kernel(sample, X[i], sigma);
    }

    std::cout << "Prediction for sample (" << sample[0] << ", " << sample[1]
              << "): " << (prediction > 0 ? 1 : -1) << "\n";
  }

  return 0;
}
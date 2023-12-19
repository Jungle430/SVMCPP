#include "CSV.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

auto CSV::loadImages(const std::string &filename) noexcept
    -> std::vector<std::vector<double>> {
  auto ans = std::vector<std::vector<double>>();
  auto file = std::ifstream(filename);
  if (file.is_open()) {
    auto line = std::string();
    while (std::getline(file, line)) {
      auto ss = std::stringstream(line);
      auto row = std::vector<double>();
      double value = 0.0;
      while (ss >> value) {
        row.emplace_back(value);
        if (ss.peek() == ',') {
          ss.ignore();
        }
      }
      ans.emplace_back(row);
    }
    file.close();
  } else {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
  }
  return ans;
}

auto CSV::loadLabels(const std::string &filename) noexcept -> std::vector<int> {
  auto ans = std::vector<int>();
  auto file = std::ifstream(filename);
  if (file.is_open()) {
    auto line = std::string();
    while (std::getline(file, line)) {
      auto ss = std::stringstream(line);
      int label = 0;
      while (ss >> label) {
      }
      ans.emplace_back(label);
    }
    file.close();
  } else {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
  }
  return ans;
}

auto CSV::writeCSV(
    const std::string &filename,
    const std::vector<std::pair<std::vector<double>, double>> &results) noexcept
    -> bool {
  std::ofstream outputFile(filename, std::ios::trunc);
  if (!outputFile.is_open()) {
    return false;
  }
  outputFile << "alpha,b" << std::endl;
  for (const auto &result : results) {
    outputFile << "\"";
    for (auto i = 0; i < result.first.size(); i++) {
      if (i == 0) {
        outputFile << result.first[i];
      } else {
        outputFile << ',' << result.first[i];
      }
    }
    outputFile << "\"," << result.second << std::endl;
  }
  outputFile.close();
  return true;
}
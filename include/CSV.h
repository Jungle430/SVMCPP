#pragma once

#include <string>
#include <utility>
#include <vector>

namespace CSV {
auto loadImages(const std::string &filename) noexcept
    -> std::vector<std::vector<double>>;

auto loadLabels(const std::string &filename) noexcept -> std::vector<int>;

auto writeCSV(
    const std::string &filename,
    const std::vector<std::pair<std::vector<double>, double>> &results) noexcept
    -> bool;
}  // namespace CSV
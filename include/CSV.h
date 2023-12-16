#pragma once

#include <string>
#include <vector>

namespace CSV {
auto loadImages(const std::string &filename) noexcept
    -> std::vector<std::vector<double>>;

auto loadLabels(const std::string &filename) noexcept -> std::vector<int>;
}  // namespace CSV
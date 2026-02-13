#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CSRProblem {
    uint64_t n{};
    uint64_t nnz{};

    std::vector<uint64_t> row_ptr;
    std::vector<uint64_t> col_idx;
    std::vector<double> values;

    std::vector<double> x;
    std::vector<double> b;
};

CSRProblem load_problem(const std::string& filename);

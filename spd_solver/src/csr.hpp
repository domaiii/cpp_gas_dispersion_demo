#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "config.hpp"
#include "fixed_point.hpp"

// ==========================
// CSRMatrix (solver matrix)
// ==========================

template<typename T>
struct CSRMatrix
{
    uint32_t n;
    uint32_t nnz;

    uint32_t row_ptr[N_MAX + 1];
    uint32_t col_idx[NNZ_MAX];
    T        values[NNZ_MAX];
};

template<typename T>
struct CSRLinearProblem
{
    CSRMatrix<T> A{};
    T x[N_MAX]{};
    T b[N_MAX]{};
};

template<typename T>
inline void csr_problem_from_file(const std::string& filename,
                                   CSRLinearProblem<T>& prob)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file");

    file.read(reinterpret_cast<char*>(&prob.A.n), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&prob.A.nnz), sizeof(uint32_t));

    if (prob.A.n > N_MAX || prob.A.nnz > NNZ_MAX)
        throw std::runtime_error("Problem exceeds static capacity");

    file.read(reinterpret_cast<char*>(prob.A.row_ptr),
              (prob.A.n + 1) * sizeof(uint32_t));

    file.read(reinterpret_cast<char*>(prob.A.col_idx),
              prob.A.nnz * sizeof(uint32_t));

    for (size_t i = 0; i < prob.A.nnz; ++i) {
        float v = 0.0f;
        file.read(reinterpret_cast<char*>(&v), sizeof(float));
        prob.A.values[i] = static_cast<T>(v);
    }

    for (size_t i = 0; i < prob.A.n; ++i) {
        float v = 0.0f;
        file.read(reinterpret_cast<char*>(&v), sizeof(float));
        prob.x[i] = static_cast<T>(v);
    }

    for (size_t i = 0; i < prob.A.n; ++i) {
        float v = 0.0f;
        file.read(reinterpret_cast<char*>(&v), sizeof(float));
        prob.b[i] = static_cast<T>(v);
    }

    if (!file)
        throw std::runtime_error("File corrupted");
}

template<typename TMatrix, typename TX, typename TY>
void spmv(const CSRMatrix<TMatrix>& A,
          const TX* x,
          TY* y)
{
    static_assert(std::is_same_v<TMatrix, TX>,
                  "spmv: matrix and x vector element types must match");
    static_assert(std::is_same_v<TMatrix, TY>,
                  "spmv: matrix and y vector element types must match");

    using T = TMatrix;

    for (size_t i = 0; i < A.n; ++i)
    {
        if constexpr (std::is_same_v<T, fixed_point>) {
            int64_t acc = 0;
            for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                acc += fixed_point::mul_wide_raw(A.values[k], x[A.col_idx[k]]);
            }
            y[i].v = static_cast<int32_t>(acc >> fixed_point::FRACTION_BITS);
        } else {
            T acc{};
            for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                acc = acc + A.values[k] * x[A.col_idx[k]];
            }
            y[i] = acc;
        }
    }
}

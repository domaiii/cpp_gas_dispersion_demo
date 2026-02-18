#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include "config.hpp"
#include "fixed_q1516.hpp"

// ==========================
// CSRMatrixStatic (solver matrix)
// ==========================

template<typename T>
struct CSRMatrixStatic
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
    CSRMatrixStatic<T> A{};
    T x[N_MAX]{};
    T b[N_MAX]{};
};

// ==========================
// CG Workspace
// ==========================

struct CGWorkspace {
    q15_16 r[N_MAX];
    q15_16 p[N_MAX];
    q15_16 Ap[N_MAX];
};

template<typename T>
inline void load_problem_from_file(const std::string& filename,
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

template<typename T>
void spmv(const CSRMatrixStatic<T>& A,
          const T* x,
          T* y)
{
    for (size_t i = 0; i < A.n; ++i)
    {
        T acc{};

        for (size_t k = A.row_ptr[i];
             k < A.row_ptr[i + 1];
             ++k)
        {
            acc = acc + A.values[k]
                        * x[A.col_idx[k]];
        }

        y[i] = acc;
    }
}

inline void spmv(const CSRMatrixStatic<q15_16>& A,
                 const q15_16* x,
                 q15_16* y)
{
    for (size_t i = 0; i < A.n; ++i)
    {
        int64_t acc = 0; 

        for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
        {
            acc += q15_16::mul_wide_raw(A.values[k], x[A.col_idx[k]]);
        }
        y[i].v = static_cast<int32_t>(acc >> 16);
    }
}

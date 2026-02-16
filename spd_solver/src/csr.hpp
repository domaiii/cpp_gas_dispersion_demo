#pragma once

#include <cstdint>
#include <string>
#include "config.hpp"
#include "fixed_q1516.hpp"

// ==========================
// CSRProblem (I/O container)
// ==========================

struct CSRProblem {
    uint32_t n = 0;
    uint32_t nnz = 0;

    uint32_t row_ptr[N_MAX + 1];
    uint32_t col_idx[NNZ_MAX];
    float values[NNZ_MAX];

    float x[N_MAX];
    float b[N_MAX];
};

void load_problem(const std::string& filename, CSRProblem& prob);

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

// ==========================
// CG Workspace
// ==========================

struct CGWorkspace {
    q15_16 r[N_MAX];
    q15_16 p[N_MAX];
    q15_16 Ap[N_MAX];
};

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
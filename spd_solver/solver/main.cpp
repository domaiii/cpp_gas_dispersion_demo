#include "csr.hpp"
#include "fixed_q1516.hpp"

#include <iostream>
#include <cmath>

static constexpr double EPS = 1e-2;

static CSRProblem prob;
static CSRMatrixStatic<q15_16> A{};
static q15_16 x[N_MAX]{};
static q15_16 y[N_MAX]{};
static q15_16 b_ref[N_MAX]{};

int main()
{
    std::cout << "=== Fixed-point SpMV demo ===\n";

    // ---------- load problem ----------

    load_problem("../spd_problem_generator/test_problems/n75_sp9.12.bin", prob);

    std::cout
        << "Loaded matrix: n="
        << prob.n
        << " nnz=" << prob.nnz
        << "\n";

    // ---------- convert to fixed CSR ----------

    A.n   = prob.n;
    A.nnz = prob.nnz;

    for (size_t i = 0; i <= prob.n; ++i)
        A.row_ptr[i] = prob.row_ptr[i];

    for (size_t i = 0; i < prob.nnz; ++i)
    {
        A.col_idx[i] = prob.col_idx[i];
        A.values[i]  = q15_16(prob.values[i]);
    }

    // ---------- convert x & reference b ----------

    for (size_t i = 0; i < prob.n; ++i)
    {
        x[i]     = q15_16(prob.x[i]);
        b_ref[i] = q15_16(prob.b[i]);
    }

    // ---------- SpMV ----------

    spmv(A, x, y);

    // ---------- compare ----------

    std::cout << "\nCompare A*x vs b:\n";

    bool ok = true;

    for (size_t i = 0; i < prob.n; ++i)
    {
        float yd = y[i].to_float();
        float bd = b_ref[i].to_float();

        float err = std::abs(yd - bd);

        if (i < 5) // print first few
        {
            std::cout
                << "i=" << i
                << "  Ax=" << yd
                << "  b="  << bd
                << "  err=" << err
                << "\n";
        }

        if (err > EPS)
            ok = false;
    }
    std::cout << "...\n";

    // ---------- result ----------

    if (ok)
        std::cout << "\nSpMV fixed-point OK)\n";
    else
        std::cout << "\nERROR: SpMV result A * x does not match vector b.\n";

    return ok ? 0 : 1;
}

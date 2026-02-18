#include "csr.hpp"
#include "fixed_q1516.hpp"

#include <iostream>
#include <cmath>

static constexpr double EPS = 1e-2;

static CSRLinearProblem<q15_16> prob{};
static q15_16 y[N_MAX]{};

int main()
{
    std::cout << "=== Fixed-point SpMV demo ===\n";

    // ---------- load problem ----------

    csr_problem_from_file("../spd_problem_generator/test_problems/n50_sp13.52.bin", prob);

    std::cout
        << "Loaded matrix: n="
        << prob.A.n
        << " nnz=" << prob.A.nnz
        << "\n";

    // ---------- SpMV ----------

    spmv(prob.A, prob.x, y);

    // ---------- compare ----------

    std::cout << "\nCompare A*x vs b:\n";

    bool ok = true;

    for (size_t i = 0; i < prob.A.n; ++i)
    {
        float yd = y[i].to_float();
        float bd = prob.b[i].to_float();

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

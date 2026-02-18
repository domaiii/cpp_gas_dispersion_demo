#include "csr.hpp"
#include "fixed_q1516.hpp"
#include "cg.hpp"

#include <iostream>
#include <cmath>

static CSRLinearProblem<fixed_point> prob{};
static CGWorkspace<fixed_point> cg_ws{};
static fixed_point x_sol[N_MAX]{};

int main()
{
    std::cout << "=== Fixed-point SpMV demo ===\n";

    // ---------- load problem ----------

    csr_problem_from_file("../spd_problem_generator/test_problems/n256_sp2.72.bin", prob);

    std::cout
        << "Loaded matrix: n="
        << prob.A.n
        << " nnz=" << prob.A.nnz
        << "\n";

    auto res = cg_solve(prob, x_sol, cg_ws);

    std::cout <<
        "=== CG Solve Test ===\n" <<
        "residual_norm = " << res.residual_norm << "\n" <<
        "iterations = " << res.iterations << "\n" <<
        "status: " << res.status << "\n";

    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    std::cout << "\nCompare solved x vs reference x:\n";
    for (size_t i = 0; i < prob.A.n; ++i) {
        const double xhat = x_sol[i].to_double();
        const double xref = prob.x[i].to_double();
        const double err = std::abs(xhat - xref);

        if (err > max_abs_err) {
            max_abs_err = err;
        }
        sum_sq_err += err * err;
        sum_sq_ref += xref * xref;

        if (i < 5) {
            std::cout
                << "i=" << i
                << "  x_sol=" << xhat
                << "  x_ref=" << xref
                << "  err=" << err
                << "\n";
        }
    }

    const double rms_err = std::sqrt(sum_sq_err / static_cast<double>(prob.A.n));
    const double rel_l2_err =
        (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;

    std::cout
        << "max_abs_err=" << max_abs_err << "\n"
        << "rms_err=" << rms_err << "\n"
        << "rel_l2_err=" << rel_l2_err << "\n";
}

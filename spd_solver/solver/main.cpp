#include "csr.hpp"
#include "fixed_point.hpp"
#include "cg.hpp"

#include <iostream>
#include <cmath>

static CSRLinearProblem<fixed_point> prob{};
static CGWorkspace<fixed_point> cg_ws{};
static fixed_point x_sol[N_MAX]{};

int main()
{
    std::cout << "=== Fixed-point CG solver ===\n";

    // ---------- load problem ----------

    csr_problem_from_file("../spd_problem_generator/test_problems/n256_sp2.72_prec.bin", prob);

    std::cout
        << "Loaded matrix: n="
        << prob.A.n
        << " nnz=" << prob.A.nnz
        << "\n";

    CGParams params{};
    // Max iteration, tolerance criteria and initial guess
    params.max_iters = N_MAX;
    params.tol = 1e-3f;
    params.zero_initial_guess = true;

    // Runtime diagnostics
    params.enable_debug_log = true;
    params.enable_bound_checks = true;
    params.direction_bound = 1.0f;

    // Optional: Search direction update and step size clipping
    params.enable_alpha_beta_clipping = false;
    params.alpha_max = 3.0f;
    params.beta_max = 1.0f;
    
    auto res = cg_solve(prob, x_sol, cg_ws, params);

    std::cout <<
        "residual_norm = " << res.residual_norm << "\n" <<
        "iterations = " << res.iterations << "\n" <<
        "status: " << res.status << "\n";

    float max_abs_err = 0.0;
    float sum_sq_err = 0.0;
    float sum_sq_ref = 0.0;

    std::cout << "\nCompare solved x vs reference x:\n";
    for (size_t i = 0; i < prob.A.n; ++i) {
        const float xhat = x_sol[i].to_float();
        const float xref = prob.x[i].to_float();
        const float err = std::abs(xhat - xref);

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

    const float rms_err = std::sqrt(sum_sq_err / static_cast<float>(prob.A.n));
    const float rel_l2_err =
        (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;

    std::cout
        << "max_abs_err=" << max_abs_err << "\n"
        << "rms_err=" << rms_err << "\n"
        << "rel_l2_err=" << rel_l2_err << "\n";
}

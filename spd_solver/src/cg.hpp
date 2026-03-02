#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <type_traits>

#include "utils.hpp"

template<typename T>
struct CGWorkspace
{
    T r[N_MAX]{};
    T p[N_MAX]{};
    T Ap[N_MAX]{};
};

struct CGParams
{
    uint32_t max_iters = static_cast<uint32_t>(N_MAX);
    float tol = 1e-3f;
    bool zero_initial_guess = true;
    bool enable_debug_log = false;
    bool enable_bound_checks = false;
    float direction_bound = 1.0f;
    bool enable_alpha_beta_clipping = false;
    float alpha_max = 3.0f;
    float beta_max = 1.0f;
    bool enable_stagnation_stop = true;
    uint32_t stagnation_window = 8;
    float min_relative_improvement = 1e-1f;
};

enum class CGStatus : uint8_t
{
    converged = 0,
    max_iters_reached = 1,
    breakdown = 2,
    stagnated = 3,
};

inline const char* to_string(CGStatus status)
{
    switch (status) {
        case CGStatus::converged:
            return "converged";
        case CGStatus::max_iters_reached:
            return "max_iters_reached";
        case CGStatus::breakdown:
            return "breakdown";
        case CGStatus::stagnated:
            return "stagnated";
        default:
            return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, CGStatus status)
{
    return os << to_string(status);
}

struct CGResult
{
    CGStatus status = CGStatus::max_iters_reached;
    uint32_t iterations = 0;
    float residual_norm = 0.0f;
};

template<typename T>
inline float cg_dot_scalar(const T* a, const T* b, uint32_t n)
{
    if constexpr (std::is_same_v<T, fixed_point>) {
        int64_t acc = 0;
        for (uint32_t i = 0; i < n; ++i) {
            acc += fixed_point::mul_wide_raw(a[i], b[i]);
        }
        const float scale = static_cast<float>(fixed_point::SCALE);
        return static_cast<float>(acc) / (scale * scale);
    } else {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            acc += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
        return acc;
    }
}

template<typename T>
inline float cg_value_to_float(T v)
{
    if constexpr (std::is_same_v<T, fixed_point>) {
        return v.to_float();
    } else {
        return static_cast<float>(v);
    }
}


template<typename T>
CGResult cg_solve(const CSRLinearProblem<T>& prob,
                  T* x,
                  CGWorkspace<T>& ws,
                  const CGParams& params = {})
{
    CGResult result{};
    const uint32_t n = prob.A.n;
    const bool debug_enabled = params.enable_debug_log;
    const bool bound_checks_enabled = params.enable_bound_checks;
    const bool collect_bounds = bound_checks_enabled || debug_enabled;
    const bool print_bound_summary = bound_checks_enabled && debug_enabled;

    if (params.zero_initial_guess) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = T{};
        }
    }

    spmv(prob.A, x, ws.Ap);
    for (uint32_t i = 0; i < n; ++i) {
        ws.r[i] = prob.b[i] - ws.Ap[i];
        ws.p[i] = ws.r[i];
    }

    float rho = cg_dot_scalar(ws.r, ws.r, n);
    float rr = rho;
    if (rr < 0.0f) {
        rr = 0.0f;
    }
    result.residual_norm = std::sqrt(rr);
    if (result.residual_norm <= params.tol) {
        result.status = CGStatus::converged;
        result.iterations = 0;
        return result;
    }

    bool direction_bound_violated = false;
    uint32_t first_violation_iter = 0;
    float max_abs_x_seen = 0.0f;
    float max_abs_r_seen = 0.0f;
    float max_abs_p_seen = 0.0f;
    float max_abs_Ap_seen = 0.0f;
    float max_abs_alpha_seen = 0.0f;
    float max_abs_beta_seen = 0.0f;
    float prev_residual_norm = result.residual_norm;
    uint32_t stagnation_count = 0;
    const auto maybe_print_bound_summary = [&]() {
        if (!print_bound_summary) return;
        std::cout
            << "[BOUND_SUMMARY] max|x|=" << max_abs_x_seen
            << " max|r|=" << max_abs_r_seen
            << " max|p|=" << max_abs_p_seen
            << " max|Ap|=" << max_abs_Ap_seen
            << " max|alpha|=" << max_abs_alpha_seen
            << " max|beta|=" << max_abs_beta_seen
            << (direction_bound_violated ? " direction_bound=ERROR" : " direction_bound=OK")
            << (direction_bound_violated ? " first_violation_iter=" : "")
            << (direction_bound_violated ? std::to_string(first_violation_iter) : "")
            << "\n";
    };

    for (uint32_t k = 0; k < params.max_iters; ++k) {
        spmv(prob.A, ws.p, ws.Ap);

        const float denom = cg_dot_scalar(ws.p, ws.Ap, n);
        if (std::abs(denom) <= 1e-20f) {
            if (debug_enabled) {
                std::cout << "[CG] breakdown at iter " << k
                          << " because denom=" << denom << "\n";
            }
            result.status = CGStatus::breakdown;
            result.iterations = k;
            return result;
        }

        float alpha = rho / denom;
        if (params.enable_alpha_beta_clipping) {
            if (alpha > params.alpha_max) alpha = params.alpha_max;
            if (alpha < -params.alpha_max) alpha = -params.alpha_max;
        }
        const float abs_alpha = std::abs(alpha);
        if (print_bound_summary && abs_alpha > max_abs_alpha_seen) max_abs_alpha_seen = abs_alpha;

        const T alpha_t = static_cast<T>(alpha);
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = x[i] + alpha_t * ws.p[i];
            ws.r[i] = ws.r[i] - alpha_t * ws.Ap[i];
        }

        const float rho_new = cg_dot_scalar(ws.r, ws.r, n);
        rr = rho_new;
        if (rr < 0.0f) {
            rr = 0.0f;
        }
        result.residual_norm = std::sqrt(rr);
        result.iterations = k + 1;
        if (result.residual_norm <= params.tol) {
            maybe_print_bound_summary();
            result.status = CGStatus::converged;
            return result;
        }
        {
            const float rel_improve =
                (prev_residual_norm - result.residual_norm) /
                std::max(prev_residual_norm, 1e-20f);
            if (rel_improve < params.min_relative_improvement) {
                ++stagnation_count;
            } else {
                stagnation_count = 0;
            }
            prev_residual_norm = result.residual_norm;
        }

        if (params.enable_stagnation_stop &&
            params.stagnation_window > 0 &&
            stagnation_count >= params.stagnation_window) {
            if (debug_enabled) {
                std::cout
                    << "[CG] stagnation stop at iter " << (k + 1)
                    << " residual_norm=" << result.residual_norm
                    << " window=" << params.stagnation_window
                    << " min_rel_improve=" << params.min_relative_improvement
                    << "\n";
            }
            result.status = CGStatus::stagnated;
            return result;
        }

        float max_abs_x = 0.0f;
        float max_abs_r = 0.0f;
        float max_abs_p = 0.0f;
        float max_abs_Ap = 0.0f;
        if (collect_bounds) {
            for (uint32_t i = 0; i < n; ++i) {
                const float ax = std::abs(cg_value_to_float(x[i]));
                const float ar = std::abs(cg_value_to_float(ws.r[i]));
                const float ap = std::abs(cg_value_to_float(ws.p[i]));
                const float aAp = std::abs(cg_value_to_float(ws.Ap[i]));
                if (ax > max_abs_x) max_abs_x = ax;
                if (ar > max_abs_r) max_abs_r = ar;
                if (ap > max_abs_p) max_abs_p = ap;
                if (aAp > max_abs_Ap) max_abs_Ap = aAp;
            }

            if (print_bound_summary) {
                if (max_abs_x > max_abs_x_seen) max_abs_x_seen = max_abs_x;
                if (max_abs_r > max_abs_r_seen) max_abs_r_seen = max_abs_r;
                if (max_abs_p > max_abs_p_seen) max_abs_p_seen = max_abs_p;
                if (max_abs_Ap > max_abs_Ap_seen) max_abs_Ap_seen = max_abs_Ap;
            }

            if (bound_checks_enabled) {
                const bool violated_now =
                    (max_abs_r > params.direction_bound) ||
                    (max_abs_p > params.direction_bound) ||
                    (max_abs_Ap > params.direction_bound);
                if (violated_now && !direction_bound_violated) {
                    direction_bound_violated = true;
                    first_violation_iter = k + 1;
                }
            }
        }

        if (debug_enabled) {
            const float beta_dbg = (std::abs(rho) > 0.0f) ? (rho_new / rho) : 0.0f;
            std::cout
                << "[CG] k=" << (k + 1)
                << " rho=" << rho
                << " denom=" << denom
                << " alpha=" << alpha
                << " beta=" << beta_dbg
                << " residual_norm=" << result.residual_norm;

            if (bound_checks_enabled) {
                std::cout
                    << " max|x|=" << max_abs_x
                    << " max|r|=" << max_abs_r
                    << " max|p|=" << max_abs_p
                    << " max|Ap|=" << max_abs_Ap
                    << (direction_bound_violated ? " bound=VIOLATED" : " bound=OK");
            }
            std::cout << "\n";
        }

        float beta = rho_new / rho;
        if (params.enable_alpha_beta_clipping) {
            if (beta > params.beta_max) beta = params.beta_max;
            if (beta < 0.0f) beta = 0.0f;
        }
        const float abs_beta = std::abs(beta);
        if (print_bound_summary && abs_beta > max_abs_beta_seen) max_abs_beta_seen = abs_beta;
        const T beta_t = static_cast<T>(beta);
        for (uint32_t i = 0; i < n; ++i) {
            ws.p[i] = ws.r[i] + beta_t * ws.p[i];
        }
        rho = rho_new;
    }

    maybe_print_bound_summary();
    result.status = CGStatus::max_iters_reached;
    return result;
}

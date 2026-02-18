#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <type_traits>

#include "csr.hpp"

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
    double tol = 1e-3;
    bool zero_initial_guess = true;
};

enum class CGStatus : uint8_t
{
    converged = 0,
    max_iters_reached = 1,
    breakdown = 2,
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
    double residual_norm = 0.0;
};

template<typename T>
inline double cg_dot_scalar(const T* a, const T* b, uint32_t n)
{
    if constexpr (std::is_same_v<T, q15_16>) {
        int64_t acc = 0;
        for (uint32_t i = 0; i < n; ++i) {
            acc += q15_16::mul_wide_raw(a[i], b[i]);
        }
        const double scale = static_cast<double>(q15_16::SCALE);
        return static_cast<double>(acc) / (scale * scale);
    } else {
        double acc = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return acc;
    }
}

template<typename T>
inline double cg_value_to_double(T v)
{
    if constexpr (std::is_same_v<T, q15_16>) {
        return v.to_double();
    } else {
        return static_cast<double>(v);
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

    double rho = cg_dot_scalar(ws.r, ws.r, n);
    double rr = rho;
    if (rr < 0.0) {
        rr = 0.0;
    }
    result.residual_norm = std::sqrt(rr);
    if (result.residual_norm <= params.tol) {
        result.status = CGStatus::converged;
        result.iterations = 0;
        return result;
    }

    for (uint32_t k = 0; k < params.max_iters; ++k) {
        spmv(prob.A, ws.p, ws.Ap);

        const double denom = cg_dot_scalar(ws.p, ws.Ap, n);
        if (std::abs(denom) <= 1e-30) {
#ifdef DEBUG_CG
            std::cout << "[DEBUG_CG] breakdown at iter " << k
                      << " because denom=" << denom << "\n";
#endif
            result.status = CGStatus::breakdown;
            result.iterations = k;
            return result;
        }

        const double alpha = rho / denom;
        for (uint32_t i = 0; i < n; ++i) {
            const double x_next =
                cg_value_to_double(x[i]) + alpha * cg_value_to_double(ws.p[i]);
            const double r_next =
                cg_value_to_double(ws.r[i]) - alpha * cg_value_to_double(ws.Ap[i]);
            x[i] = static_cast<T>(x_next);
            ws.r[i] = static_cast<T>(r_next);
        }

        const double rho_new = cg_dot_scalar(ws.r, ws.r, n);
        rr = rho_new;
        if (rr < 0.0) {
            rr = 0.0;
        }
        result.residual_norm = std::sqrt(rr);
        result.iterations = k + 1;
#ifdef DEBUG_CG
        if (k < 10 || ((k + 1) % 20 == 0)) {
            double max_abs_x = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const double ax = std::abs(cg_value_to_double(x[i]));
                if (ax > max_abs_x) max_abs_x = ax;
            }
            std::cout
                << "[DEBUG_CG] k=" << (k + 1)
                << " rho=" << rho
                << " denom=" << denom
                << " alpha=" << alpha
                << " beta=" << ((std::abs(rho) > 0.0) ? (rho_new / rho) : 0.0)
                << " residual_norm=" << result.residual_norm
                << " max|x|=" << max_abs_x << "\n";
        }
#endif

        if (result.residual_norm <= params.tol) {
            result.status = CGStatus::converged;
            return result;
        }

        const double beta = rho_new / rho;
        for (uint32_t i = 0; i < n; ++i) {
            const double p_next =
                cg_value_to_double(ws.r[i]) + beta * cg_value_to_double(ws.p[i]);
            ws.p[i] = static_cast<T>(p_next);
        }
        rho = rho_new;
    }

    result.status = CGStatus::max_iters_reached;
    return result;
}

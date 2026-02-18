#pragma once

#include <cmath>
#include <cstdint>
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
inline T cg_dot(const T* a, const T* b, uint32_t n)
{
    if constexpr (std::is_same_v<T, q15_16>) {
        return dot_q15_16(a, b, n);
    } else {
        T acc{};
        for (uint32_t i = 0; i < n; ++i) {
            acc = acc + a[i] * b[i];
        }
        return acc;
    }
}

template<typename T>
inline double cg_to_double(T v)
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

    T rho = cg_dot(ws.r, ws.r, n);
    double rr = cg_to_double(rho);
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

        const T denom = cg_dot(ws.p, ws.Ap, n);
        if (std::abs(cg_to_double(denom)) <= 1e-30) {
            result.status = CGStatus::breakdown;
            result.iterations = k;
            return result;
        }

        const T alpha = rho / denom;
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = x[i] + alpha * ws.p[i];
            ws.r[i] = ws.r[i] - alpha * ws.Ap[i];
        }

        const T rho_new = cg_dot(ws.r, ws.r, n);
        rr = cg_to_double(rho_new);
        if (rr < 0.0) {
            rr = 0.0;
        }
        result.residual_norm = std::sqrt(rr);
        result.iterations = k + 1;

        if (result.residual_norm <= params.tol) {
            result.status = CGStatus::converged;
            return result;
        }

        const T beta = rho_new / rho;
        for (uint32_t i = 0; i < n; ++i) {
            ws.p[i] = ws.r[i] + beta * ws.p[i];
        }
        rho = rho_new;
    }

    result.status = CGStatus::max_iters_reached;
    return result;
}

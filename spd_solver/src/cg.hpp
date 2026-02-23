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
    float tol = 1e-3f;
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

#ifdef DEBUG_CG
    constexpr float kDirectionBound = 1.0f;
    bool direction_bound_violated = false;
    uint32_t first_violation_iter = 0;
    float max_abs_x_seen = 0.0f;
    float max_abs_r_seen = 0.0f;
    float max_abs_p_seen = 0.0f;
    float max_abs_Ap_seen = 0.0f;
    float max_abs_alpha_seen = 0.0f;
    float max_abs_beta_seen = 0.0f;
#endif

    for (uint32_t k = 0; k < params.max_iters; ++k) {
        spmv(prob.A, ws.p, ws.Ap);

        const float denom = cg_dot_scalar(ws.p, ws.Ap, n);
        if (std::abs(denom) <= 1e-20f) {
#ifdef DEBUG_CG
            std::cout << "[DEBUG_CG] breakdown at iter " << k
                      << " because denom=" << denom << "\n";
            std::cout
                << "[BOUND_SUMMARY] max|x|=" << max_abs_x_seen
                << " max|r|=" << max_abs_r_seen
                << " max|p|=" << max_abs_p_seen
                << " max|Ap|=" << max_abs_Ap_seen
                << " max|alpha|=" << max_abs_alpha_seen
                << " max|beta|=" << max_abs_beta_seen
                << (direction_bound_violated ? " direction_bound=VIOLATED" : " direction_bound=OK")
                << (direction_bound_violated ? " first_violation_iter=" : "")
                << (direction_bound_violated ? std::to_string(first_violation_iter) : "")
                << "\n";
#endif
            result.status = CGStatus::breakdown;
            result.iterations = k;
            return result;
        }

        const float alpha = rho / denom;
#ifdef DEBUG_CG
        {
            const float abs_alpha = std::abs(alpha);
            if (abs_alpha > max_abs_alpha_seen) max_abs_alpha_seen = abs_alpha;
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            const float x_next =
                cg_value_to_float(x[i]) + alpha * cg_value_to_float(ws.p[i]);
            const float r_next =
                cg_value_to_float(ws.r[i]) - alpha * cg_value_to_float(ws.Ap[i]);
            x[i] = static_cast<T>(x_next);
            ws.r[i] = static_cast<T>(r_next);
        }

        const float rho_new = cg_dot_scalar(ws.r, ws.r, n);
        rr = rho_new;
        if (rr < 0.0f) {
            rr = 0.0f;
        }
        result.residual_norm = std::sqrt(rr);
        result.iterations = k + 1;
#ifdef DEBUG_CG
        if (k < 10 || ((k + 1) % 20 == 0)) {
            float max_abs_x = 0.0f;
            float max_abs_r = 0.0f;
            float max_abs_p = 0.0f;
            float max_abs_Ap = 0.0f;
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

            if (max_abs_x > max_abs_x_seen) max_abs_x_seen = max_abs_x;
            if (max_abs_r > max_abs_r_seen) max_abs_r_seen = max_abs_r;
            if (max_abs_p > max_abs_p_seen) max_abs_p_seen = max_abs_p;
            if (max_abs_Ap > max_abs_Ap_seen) max_abs_Ap_seen = max_abs_Ap;

            const bool violated_now =
                (max_abs_r > kDirectionBound) ||
                (max_abs_p > kDirectionBound) ||
                (max_abs_Ap > kDirectionBound);
            if (violated_now && !direction_bound_violated) {
                direction_bound_violated = true;
                first_violation_iter = k + 1;
            }

            std::cout
                << "[DEBUG_CG] k=" << (k + 1)
                << " rho=" << rho
                << " denom=" << denom
                << " alpha=" << alpha
                << " beta=" << ((std::abs(rho) > 0.0f) ? (rho_new / rho) : 0.0f)
                << " residual_norm=" << result.residual_norm
                << " max|x|=" << max_abs_x
                << " max|r|=" << max_abs_r
                << " max|p|=" << max_abs_p
                << " max|Ap|=" << max_abs_Ap
                << (direction_bound_violated ? " bound=VIOLATED" : " bound=OK")
                << "\n";
        }
#endif

        if (result.residual_norm <= params.tol) {
#ifdef DEBUG_CG
            std::cout
                << "[BOUND_SUMMARY] max|x|=" << max_abs_x_seen
                << " max|r|=" << max_abs_r_seen
                << " max|p|=" << max_abs_p_seen
                << " max|Ap|=" << max_abs_Ap_seen
                << " max|alpha|=" << max_abs_alpha_seen
                << " max|beta|=" << max_abs_beta_seen
                << (direction_bound_violated ? " direction_bound=VIOLATED" : " direction_bound=OK")
                << (direction_bound_violated ? " first_violation_iter=" : "")
                << (direction_bound_violated ? std::to_string(first_violation_iter) : "")
                << "\n";
#endif
            result.status = CGStatus::converged;
            return result;
        }

        const float beta = rho_new / rho;
#ifdef DEBUG_CG
        {
            const float abs_beta = std::abs(beta);
            if (abs_beta > max_abs_beta_seen) max_abs_beta_seen = abs_beta;
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            const float p_next =
                cg_value_to_float(ws.r[i]) + beta * cg_value_to_float(ws.p[i]);
            ws.p[i] = static_cast<T>(p_next);
        }
        rho = rho_new;
    }

#ifdef DEBUG_CG
    std::cout
        << "[BOUND_SUMMARY] max|x|=" << max_abs_x_seen
        << " max|r|=" << max_abs_r_seen
        << " max|p|=" << max_abs_p_seen
        << " max|Ap|=" << max_abs_Ap_seen
        << " max|alpha|=" << max_abs_alpha_seen
        << " max|beta|=" << max_abs_beta_seen
        << (direction_bound_violated ? " direction_bound=VIOLATED" : " direction_bound=OK")
        << (direction_bound_violated ? " first_violation_iter=" : "")
        << (direction_bound_violated ? std::to_string(first_violation_iter) : "")
        << "\n";
#endif
    result.status = CGStatus::max_iters_reached;
    return result;
}

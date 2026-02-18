#pragma once

#include <iostream>
#include <cstdint>
#include <cmath>
#include "config.hpp"

// =============================================================================
// Core Fixed-Point Type
// =============================================================================

struct fixed_point {
    int32_t v;

    static constexpr int FRACTION_BITS = FIXED_FRACTION_BITS;
    static constexpr int32_t SCALE = 1 << FRACTION_BITS;

    fixed_point() : v(0) {}
    explicit fixed_point(float f) : fixed_point(static_cast<double>(f)) {}
    explicit fixed_point(double d)
    {
        if (!std::isfinite(d)) {
            v = 0;
            return;
        }

        constexpr double maxv = static_cast<double>(INT32_MAX) /
                                static_cast<double>(SCALE);
        constexpr double minv = static_cast<double>(INT32_MIN) /
                                static_cast<double>(SCALE);

        if (d > maxv) d = maxv;
        if (d < minv) d = minv;

        const double scaled = d * static_cast<double>(SCALE);
        int64_t raw = static_cast<int64_t>(std::llround(scaled));

        if (raw > INT32_MAX) raw = INT32_MAX;
        if (raw < INT32_MIN) raw = INT32_MIN;

        v = static_cast<int32_t>(raw);
    }

    static fixed_point from_raw(int32_t raw) { fixed_point x; x.v = raw; return x; }

    double to_double() const { return static_cast<double>(v) / static_cast<double>(SCALE); }
    float to_float() const { return static_cast<float>(v) / static_cast<float>(SCALE); }

    // Overload operators + - * /
    fixed_point operator+(const fixed_point& o) const {
        int64_t tmp = static_cast<int64_t>(v) + static_cast<int64_t>(o.v);
        if (tmp > INT32_MAX) tmp = INT32_MAX;
        if (tmp < INT32_MIN) tmp = INT32_MIN;
        return from_raw(static_cast<int32_t>(tmp));
    }

    fixed_point operator-(const fixed_point& o) const {
        int64_t tmp = static_cast<int64_t>(v) - static_cast<int64_t>(o.v);
        if (tmp > INT32_MAX) tmp = INT32_MAX;
        if (tmp < INT32_MIN) tmp = INT32_MIN;
        return from_raw(static_cast<int32_t>(tmp));
    }

    fixed_point operator*(const fixed_point& o) const {
        int64_t prod = static_cast<int64_t>(v) * static_cast<int64_t>(o.v);
        int64_t shifted = prod >> FRACTION_BITS;
        if (shifted > INT32_MAX) shifted = INT32_MAX;
        if (shifted < INT32_MIN) shifted = INT32_MIN;
        return from_raw(static_cast<int32_t>(shifted));
    }

    fixed_point operator/(const fixed_point& o) const {
        if (o.v == 0) {
            return from_raw(v >= 0 ? INT32_MAX : INT32_MIN);
        }

        int64_t num = static_cast<int64_t>(v) << FRACTION_BITS;
        int64_t raw = num / static_cast<int64_t>(o.v);

        if (raw > INT32_MAX) raw = INT32_MAX;
        if (raw < INT32_MIN) raw = INT32_MIN;

        return from_raw(static_cast<int32_t>(raw));
    }

    static int64_t mul_wide_raw(const fixed_point& a, const fixed_point& b) {
        return static_cast<int64_t>(a.v) * static_cast<int64_t>(b.v);
    }

    static fixed_point from_accum_q30_32(int64_t accum_q30_32) {
        int64_t raw = accum_q30_32 >> FRACTION_BITS;
        if (raw > INT32_MAX) raw = INT32_MAX;
        if (raw < INT32_MIN) raw = INT32_MIN;
        return from_raw(static_cast<int32_t>(raw));
    }
};

// =============================================================================
// Vector Operations (Helper Functions)
// =============================================================================

// dot(a,b) with 64-bit accumulation
static inline fixed_point dot_fixed(const fixed_point* a, const fixed_point* b, uint32_t n) {
    int64_t acc = 0;
    for (size_t i = 0; i < n; ++i)
        acc += fixed_point::mul_wide_raw(a[i], b[i]);
    return fixed_point::from_accum_q30_32(acc);
}

// copy vector
static inline void vec_copy(fixed_point* dst, const fixed_point* src, uint32_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

// y += alpha * x
static inline void vec_axpy(fixed_point* y, const fixed_point* x, fixed_point alpha, uint32_t n) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] + alpha * x[i];
}

// z = x + alpha * y (often used as p = r + beta * p)
static inline void vec_xpay(fixed_point* p, const fixed_point* r, fixed_point beta, uint32_t n) {
    for (size_t i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
}

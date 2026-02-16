#pragma once

#include <iostream>
#include <cstdint>
#include <cmath>

// =============================================================================
// Core Fixed-Point Type
// =============================================================================

struct q15_16 {
    int32_t v;

    static constexpr int FRACTION_BITS = 16;
    static constexpr int32_t SCALE = 1 << FRACTION_BITS;

    q15_16() : v(0) {}
    explicit q15_16(float f) 
        : v(static_cast<int32_t>(f * static_cast<float>(SCALE))) {}

    static q15_16 from_raw(int32_t raw) { q15_16 x; x.v = raw; return x; }

    double to_double() const { return static_cast<double>(v) / static_cast<double>(SCALE); }
    float to_float() const { return static_cast<float>(v) / static_cast<float>(SCALE); }

    // Overload operators + - * /
    q15_16 operator+(const q15_16& o) const {
        return from_raw(v + o.v);
    }

    q15_16 operator-(const q15_16& o) const {
        return from_raw(v - o.v);
    }

    q15_16 operator*(const q15_16& o) const {
        int64_t tmp = static_cast<int64_t>(v) * o.v;
        return from_raw(static_cast<int32_t>(tmp >> FRACTION_BITS));
    }

    q15_16 operator/(const q15_16& o) const {
        if (o.v == 0) return from_raw(v >= 0 ? 0x7FFFFFFF : 0x80000000);
        int64_t num = static_cast<int64_t>(v) << FRACTION_BITS;
        return from_raw(static_cast<int32_t>(num / o.v));
    }

    static int64_t mul_wide_raw(const q15_16& a, const q15_16& b) {
        return static_cast<int64_t>(a.v) * static_cast<int64_t>(b.v);
    }

    static q15_16 from_accum_q30_32(int64_t accum_q30_32) {
        return from_raw(static_cast<int32_t>(accum_q30_32 >> FRACTION_BITS));
    }
};

// =============================================================================
// Vector Operations (Helper Functions)
// =============================================================================

// dot(a,b) with 64-bit accumulation
static inline q15_16 dot_q15_16(const q15_16* a, const q15_16* b, uint32_t n) {
    int64_t acc = 0;
    for (size_t i = 0; i < n; ++i)
        acc += q15_16::mul_wide_raw(a[i], b[i]);
    return q15_16::from_accum_q30_32(acc);
}

// copy vector
static inline void vec_copy(q15_16* dst, const q15_16* src, uint32_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

// y += alpha * x
static inline void vec_axpy(q15_16* y, const q15_16* x, q15_16 alpha, uint32_t n) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] + alpha * x[i];
}

// z = x + alpha * y (often used as p = r + beta * p)
static inline void vec_xpay(q15_16* p, const q15_16* r, q15_16 beta, uint32_t n) {
    for (size_t i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
}
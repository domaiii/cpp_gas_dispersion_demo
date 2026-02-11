#define EIGEN_RUNTIME_NO_MALLOC

#include <iostream>
#include <cmath>
#include <cstddef>
#include <chrono>

static constexpr int NX = 75; // cells in x
static constexpr int NY = 75; // cells in y

// Interior unknowns
static constexpr int NXI = NX - 1;
static constexpr int NYI = NY - 1;
static constexpr int N   = NXI * NYI;

static constexpr float h   = 1.0 / float(NX);
static constexpr float ih2 = 1.0 / (h * h);

static float x[N];   // solution
static float b[N];   // RHS
static float r[N];   // CG residual
static float p[N];   // CG search direction
static float Ap[N];  // CG vector

static inline float dot(const float* a, const float* c) {
    float s = 0.0;
    for (int i = 0; i < N; ++i) s += a[i] * c[i];
    return s;
}

static inline float norm2(const float* a) {
    return std::sqrt(dot(a, a));
}

static inline void axpy(float* y, float alpha, const float* xvec) {
    for (int i = 0; i < N; ++i) y[i] += alpha * xvec[i];
}

static inline void copy(float* dst, const float* src) {
    for (int i = 0; i < N; ++i) dst[i] = src[i];
}

static inline int k_of(int i, int j) {
    return (j - 1) * NXI + (i - 1);
}

// Matrix-free apply: y = A x, where A corresponds to stiffness (Laplacian)
static void apply_A(const float* xvec, float* yvec) {
    for (int j = 1; j <= NY - 1; ++j) {
        for (int i = 1; i <= NX - 1; ++i) {
            const int k = k_of(i, j);

            const float xc = xvec[k];

            // Neighbor values in interior vector; boundary outside interior => 0
            const float xl = (i > 1)      ? xvec[k_of(i - 1, j)] : 0.0;
            const float xr = (i < NX - 1) ? xvec[k_of(i + 1, j)] : 0.0;
            const float xd = (j > 1)      ? xvec[k_of(i, j - 1)] : 0.0;
            const float xu = (j < NY - 1) ? xvec[k_of(i, j + 1)] : 0.0;

            // SPD stiffness operator
            yvec[k] = ih2 * (4.0 * xc - xl - xr - xd - xu);
        }
    }
}

int main() {
    // Matrix free CG solver
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        x[i]  = 0.0;
        b[i]  = 1.0;
        r[i]  = 0.0;
        p[i]  = 0.0;
        Ap[i] = 0.0;
    }

    // r0 = b - A*x (x=0 => r=b)
    copy(r, b);
    copy(p, r);

    const float bnorm = norm2(b);
    if (bnorm == 0.0) {
        std::cout << "bnorm=0, trivial.\n";
        return 0;
    }

    float rsold = dot(r, r);

    const int maxit = 2000;
    const float tol_rel = 1e-8;

    int it = 0;
    for (; it < maxit; ++it) {
        apply_A(p, Ap);

        const float pAp = dot(p, Ap);
        if (pAp <= 0.0) {
            std::cout << "Breakdown: p^T A p <= 0 (pAp=" << pAp << ")\n";
            break;
        }

        const float alpha = rsold / pAp;

        // x = x + alpha p
        axpy(x, alpha, p);

        // r = r - alpha Ap
        axpy(r, -alpha, Ap);

        const float rsnew = dot(r, r);

        const float rel = std::sqrt(rsnew) / bnorm;
        if (rel < tol_rel) {
            rsold = rsnew;
            ++it;
            break;
        }

        const float beta = rsnew / rsold;

        // p = r + beta p
        for (int k = 0; k < N; ++k) p[k] = r[k] + beta * p[k];

        rsold = rsnew;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fs = t1 - t0;
    std::cout << "CG iters: " << it << "\n"
              << "rel_res: " << (std::sqrt(rsold) / bnorm) << "\n"
              << "Time: " << fs.count() << " s" << "\n";

    const int ic = NX / 2;
    const int jc = NY / 2;
    const int kc = k_of(ic, jc);
    std::cout << "u(center) = " << x[kc] << "\n";

    return 0;
}

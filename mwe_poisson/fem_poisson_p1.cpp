#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "mesh.hpp"

static inline double tri_area(const Node& a, const Node& b, const Node& c) {
    return 0.5 * std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

// ∫_T grad(phi_i)·grad(phi_j) dΩ = (b_i b_j + c_i c_j) / (4A)
static std::array<std::array<double,3>,3>
local_stiffness_p1(const Node& p0, const Node& p1, const Node& p2) {
    double x0 = p0.x, y0 = p0.y;
    double x1 = p1.x, y1 = p1.y;
    double x2 = p2.x, y2 = p2.y;

    double det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    double A = 0.5 * std::abs(det);
    double scale = 1.0 / (4.0 * A);

    double b0 = y1 - y2, c0 = x2 - x1;
    double b1 = y2 - y0, c1 = x0 - x2;
    double b2 = y0 - y1, c2 = x1 - x0;

    std::array<double,3> b{b0, b1, b2};
    std::array<double,3> c{c0, c1, c2};

    std::array<std::array<double,3>,3> K{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K[i][j] = scale * (b[i]*b[j] + c[i]*c[j]);

    return K;
}

// f=1 constant: ∫_T f φ_i dΩ = area/3
static inline std::array<double,3> local_load_constant_f(double area, double fval = 1.0) {
    return {fval * area / 3.0, fval * area / 3.0, fval * area / 3.0};
}

int main() {
    // Problem size
    int nx = 500, ny = 500;

    Mesh mesh = make_unit_square_mesh(nx, ny);
    const int n_nodes = (int)mesh.nodes.size();

    // Map global node -> unknown index (only interior nodes get an index)
    std::vector<int> g2u(n_nodes, -1);
    int n_unk = 0;
    for (int i = 0; i < n_nodes; ++i) {
        if (!mesh.is_dirichlet[i]) {
            g2u[i] = n_unk++;
        }
    }

    using SpMat = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    std::vector<Triplet> trips;
    trips.reserve(mesh.tris.size() * 9); // upper bound for interior contributions

    Eigen::VectorXd b = Eigen::VectorXd::Zero(n_unk);

    // Assemble reduced system (interior unknowns only). Dirichlet is u=0, so no RHS correction.
    for (const auto& t : mesh.tris) {
        const Node& p0 = mesh.nodes[t.v[0]];
        const Node& p1 = mesh.nodes[t.v[1]];
        const Node& p2 = mesh.nodes[t.v[2]];

        double area = tri_area(p0, p1, p2);
        auto K = local_stiffness_p1(p0, p1, p2);
        auto f = local_load_constant_f(area, 1.0);

        for (int a = 0; a < 3; ++a) {
            int ga = t.v[a];
            int ia = g2u[ga];

            // Load vector: only if test function corresponds to an interior unknown
            if (ia >= 0) {
                b[ia] += f[a];
            }

            for (int c = 0; c < 3; ++c) {
                int gc = t.v[c];
                int ic = g2u[gc];

                // Stiffness contributions:
                // If both are interior -> matrix entry.
                // If basis is Dirichlet and Dirichlet value is zero -> no RHS correction needed.
                if (ia >= 0 && ic >= 0) {
                    trips.emplace_back(ia, ic, K[a][c]);
                }
            }
        }
    }

    SpMat A(n_unk, n_unk);
    A.setFromTriplets(trips.begin(), trips.end()); // duplicates summed

    // Solve A x = b with PCG (SPD)
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;
    cg.setMaxIterations(2000);
    cg.setTolerance(1e-8);
    cg.compute(A);

    Eigen::VectorXd x_unk = cg.solve(b);

    std::cout << "CG iters: " << cg.iterations()
              << "  est_relerr: " << cg.error() << "\n";

    // Expand solution to all nodes (Dirichlet nodes = 0)
    std::vector<double> x_full(n_nodes, 0.0);
    for (int g = 0; g < n_nodes; ++g) {
        int u = g2u[g];
        if (u >= 0) x_full[g] = x_unk[u];
    }

    // Print center value (closest grid node)
    int cx = nx / 2, cy = ny / 2;
    int center = cy * (nx + 1) + cx;
    std::cout << "u(center) CG = " << x_full[center] << "\n";

    // Reference with direct Cholesky
    Eigen::SimplicialLDLT<SpMat> chol;
    chol.compute(A);
    auto x_chol = chol.solve(b);
    std::cout << "u(center) Cholesky = " << x_chol[center] << "\n";

    // std::cout << Eigen::MatrixXd(A) << std::endl;

    return 0;
}

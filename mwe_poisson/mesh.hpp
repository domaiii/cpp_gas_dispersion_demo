#pragma once
#include <array>
#include <vector>

struct Node { double x, y; };

struct Tri { std::array<int, 3> v; };

struct Mesh {
    std::vector<Node> nodes;
    std::vector<Tri>  tris;
    std::vector<unsigned char> is_dirichlet; // 1 if boundary node
};

inline Mesh make_unit_square_mesh(int nx, int ny) {
    Mesh m;
    const int npx = nx + 1;
    const int npy = ny + 1;

    m.nodes.reserve(npx * npy);
    m.is_dirichlet.assign(npx * npy, 0);

    for (int j = 0; j < npy; ++j) {
        for (int i = 0; i < npx; ++i) {
            double x = double(i) / nx;
            double y = double(j) / ny;
            m.nodes.push_back({x, y});
            if (i == 0 || j == 0 || i == nx || j == ny) {
                m.is_dirichlet[j * npx + i] = 1;
            }
        }
    }

    auto vid = [&](int i, int j) { return j * npx + i; };

    m.tris.reserve(2 * nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int v00 = vid(i, j);
            int v10 = vid(i + 1, j);
            int v01 = vid(i, j + 1);
            int v11 = vid(i + 1, j + 1);
            m.tris.push_back({{v00, v10, v11}});
            m.tris.push_back({{v00, v11, v01}});
        }
    }
    return m;
}
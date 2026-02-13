#include "csr_problem_reader.hpp"

#include <fstream>
#include <stdexcept>

CSRProblem load_problem(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);

    CSRProblem prob{};

    // --- Read header ---
    file.read(reinterpret_cast<char*>(&prob.n), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&prob.nnz), sizeof(uint64_t));

    if (!file)
        throw std::runtime_error("Failed reading header.");

    if (prob.n == 0)
        throw std::runtime_error("Invalid matrix size (n = 0).");

    // --- Allocate ---
    prob.row_ptr.resize(prob.n + 1);
    prob.col_idx.resize(prob.nnz);
    prob.values.resize(prob.nnz);
    prob.x.resize(prob.n);
    prob.b.resize(prob.n);

    // --- Read CSR data ---
    file.read(reinterpret_cast<char*>(prob.row_ptr.data()),
              (prob.n + 1) * sizeof(uint64_t));

    file.read(reinterpret_cast<char*>(prob.col_idx.data()),
              prob.nnz * sizeof(uint64_t));

    file.read(reinterpret_cast<char*>(prob.values.data()),
              prob.nnz * sizeof(double));

    // --- Read vectors ---
    file.read(reinterpret_cast<char*>(prob.x.data()),
              prob.n * sizeof(double));

    file.read(reinterpret_cast<char*>(prob.b.data()),
              prob.n * sizeof(double));

    if (!file)
        throw std::runtime_error("File corrupted or incomplete.");

    // --- Basic CSR validation ---
    if (prob.row_ptr[0] != 0)
        throw std::runtime_error("CSR row_ptr[0] must be 0.");

    if (prob.row_ptr[prob.n] != prob.nnz)
        throw std::runtime_error("CSR row_ptr[n] must equal nnz.");

    return prob;
}

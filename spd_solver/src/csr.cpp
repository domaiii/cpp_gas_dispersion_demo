#include "csr.hpp"
#include "fixed_q1516.hpp"
#include <fstream>
#include <stdexcept>

void load_problem(const std::string& filename, CSRProblem& prob)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file");

    file.read((char*)&prob.n, sizeof(uint32_t));
    file.read((char*)&prob.nnz, sizeof(uint32_t));

    if (prob.n > N_MAX || prob.nnz > NNZ_MAX)
        throw std::runtime_error("Problem exceeds static capacity");

    file.read((char*)prob.row_ptr,
              (prob.n + 1)*sizeof(uint32_t));

    file.read((char*)prob.col_idx,
              prob.nnz*sizeof(uint32_t));

    file.read((char*)prob.values,
              prob.nnz*sizeof(float));

    file.read((char*)prob.x,
              prob.n*sizeof(float));

    file.read((char*)prob.b,
              prob.n*sizeof(float));

    if (!file)
        throw std::runtime_error("File corrupted");
}

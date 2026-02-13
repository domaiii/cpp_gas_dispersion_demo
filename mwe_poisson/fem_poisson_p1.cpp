/* #include <cnl/all.h>
#include <iostream>

using cnl::power;
using cnl::scaled_integer;

int main() {
    auto a = scaled_integer<int32_t, power<-8>>{2.5};
    auto b = scaled_integer<int32_t, power<-8>>{1.5};
    
    auto res = a * b;
    std::cout << res << '\n';
    return 0;
} */

#include "csr_problem_reader.hpp"
#include <iostream>

int main() {
    try {
        auto problem = load_problem("../matrix_generator/test_problems/n25_sp19.04.bin");

        std::cout << "Loaded problem:\n";
        std::cout << "n   = " << problem.n << "\n";
        std::cout << "nnz = " << problem.nnz << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
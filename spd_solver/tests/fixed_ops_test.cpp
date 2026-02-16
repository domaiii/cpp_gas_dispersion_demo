// #define DEBUG_FIXED 
#include "fixed_q1516.hpp"
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cassert>
#include <climits>

static constexpr double EPS = 1e-3;

void check_close(double ref, double got, const char* msg) {
    std::cout << std::left << std::setw(28) << msg << "OK  "
              << "ref=" << std::setw(8) << ref 
              << " got=" << std::setw(8) << got 
              << " err=" << std::abs(ref - got) << "\n";
    assert(std::abs(ref - got) < EPS);
}

void check_ok(const char* msg, const char* extra = "") {
    std::cout << std::left << std::setw(28) << msg << "OK  " << extra << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== q15_16 arithmetic test ===\n";

    // ---------- conversion ----------
    {
        double x = 1.25;
        q15_16 q(x);
        check_close(x, q.to_double(), "Conversion test");
    }

    // ---------- add/sub ----------
    {
        q15_16 a(1.5), b(0.25);
        check_close(1.75, (a+b).to_double(), "Addition test");
        check_close(1.25, (a-b).to_double(), "Subtraction test");
    }

    // ---------- multiply ----------
    {
        q15_16 a(1.5), b(0.5);
        check_close(0.75, (a*b).to_double(), "Multiplication test");
    }

    // ---------- division ----------
    {
        q15_16 a(1.5), b(0.5);
        check_close(3.0, (a/b).to_double(), "Division test");
    }

    // ---------- negative division ----------
    {
        q15_16 a(-2.0), b(0.5);
        check_close(-4.0, (a/b).to_double(), "Negative div test");
    }

    // ---------- dot product ----------
    {
        q15_16 x[3] = { q15_16(1.0), q15_16(2.0), q15_16(3.0) };
        q15_16 d = dot_q15_16(x, x, 3);
        check_close(14.0, d.to_double(), "Dot product test");
    }

    // ---------- axpy ----------
    {
        q15_16 y[3] = { q15_16(1.0), q15_16(1.0), q15_16(1.0) };
        q15_16 x[3] = { q15_16(1.0), q15_16(2.0), q15_16(3.0) };
        q15_16 alpha(2.0);

        vec_axpy(y, x, alpha, 3);

        check_close(3.0, y[0].to_double(), "AXPY function test");
        check_close(5.0, y[1].to_double(), "");
        check_close(7.0, y[2].to_double(), "");
    }

    std::cout << "\n=== Overflow tests ===\n";

    // ---------- addition overflow (Wraparound!) ----------
    {
        q15_16 a = q15_16::from_raw(INT32_MAX); 
        q15_16 b = q15_16::from_raw(1);
        auto r = a + b;
        assert(r.v == INT32_MIN);
        check_ok("Addition Overflow");
    }

    // ---------- multiplication overflow ----------
    {
        float big_val = 200.0f;
        q15_16 a(big_val);
        q15_16 b(big_val);
        q15_16 r_fixed = a * b; 

        double r_true = (double)big_val * big_val;
        assert(std::abs(r_true - r_fixed.to_double()) > 1.0);

        int64_t full_res = ((int64_t)a.v * b.v) >> 16;
        int32_t expected_wrap = (int32_t)full_res;
        assert(r_fixed.v == expected_wrap);
        
        check_ok("Multiplication Overflow");
    }

    // ---------- division overflow (Saturation) ----------
    {
        double val_a = 1000.0;
        double val_b = 0.0001;
        q15_16 a(val_a);
        q15_16 b(val_b);
        auto r = a / b;

        double math_result = val_a / val_b;
        assert(std::abs(math_result - r.to_double()) > 1.0); 
        
        check_ok("Division overflow");
    }

    // ---------- division by zero ----------
    {
        q15_16 a(1.0);
        q15_16 zero;
        auto r = a / zero;
        assert(r.v == INT32_MAX);
        check_ok("Division by zero:", "(saturated to MAX)");
    }

    // ---------- dot accumulation stress ----------
    {
        q15_16 x[8];
        int64_t manual_acc = 0;
        for(int i=0; i<8; i++) {
            x[i] = q15_16(100.0);
            manual_acc += ((int64_t)x[i].v * x[i].v) >> 16;
        }
        q15_16 d = dot_q15_16(x, x, 8);
        assert(d.v == (int32_t)manual_acc);
        check_ok("Dot product overflow");
    }

    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}
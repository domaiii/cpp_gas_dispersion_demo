// #define DEBUG_FIXED 
#include "fixed_point.hpp"
#include "utils.hpp"
#include "cg.hpp"
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cassert>
#include <climits>
#include <filesystem>

static constexpr float EPS = 1e-3;
static CSRLinearProblem<fixed_point> prob{};
static fixed_point y[N_MAX]{};

void check_close(float ref, float got, const char* msg) {
    assert(std::abs(ref - got) < EPS);
    std::cout << std::left << std::setw(28) << msg << "OK  "
              << "ref=" << std::setw(8) << ref 
              << " got=" << std::setw(8) << got 
              << " err=" << std::abs(ref - got) << "\n";
    
}

void check_ok(const char* msg, const char* extra = "") {
    std::cout << std::left << std::setw(28) << msg << "OK  " << extra << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== fixed_point arithmetic test ===\n";

    // ---------- conversion ----------
    {
        float x = 1.25;
        fixed_point q(x);
        check_close(x, q.to_float(), "Conversion test");
    }

    // ---------- add/sub ----------
    {
        fixed_point a(1.5), b(0.25);
        check_close(1.75, (a+b).to_float(), "Addition test");
        check_close(1.25, (a-b).to_float(), "Subtraction test");
    }

    // ---------- multiply ----------
    {
        fixed_point a(1.5), b(0.5);
        check_close(0.75, (a*b).to_float(), "Multiplication test");
    }

    // ---------- division ----------
    {
        fixed_point a(1.5), b(0.5);
        check_close(3.0, (a/b).to_float(), "Division test");
    }

    // ---------- negative division ----------
    {
        fixed_point a(-2.0), b(0.5);
        check_close(-4.0, (a/b).to_float(), "Negative div test");
    }

    // ---------- dot product ----------
    {
        fixed_point x[3] = { fixed_point(1.0), fixed_point(2.0), fixed_point(3.0) };
        fixed_point d = dot_fixed(x, x, 3);
        check_close(14.0, d.to_float(), "Dot product test");
    }

    // ---------- axpy ----------
    {
        fixed_point y[3] = { fixed_point(1.0), fixed_point(1.0), fixed_point(1.0) };
        fixed_point x[3] = { fixed_point(1.0), fixed_point(2.0), fixed_point(3.0) };
        fixed_point alpha(2.0);

        vec_axpy(y, x, alpha, 3);

        check_close(3.0, y[0].to_float(), "AXPY function test");
        check_close(5.0, y[1].to_float(), "---");
        check_close(7.0, y[2].to_float(), "---");
    }

    std::cout << "\n=== Overflow tests ===\n";

    // ---------- addition overflow (Saturation) ----------
    {
        fixed_point a = fixed_point::from_raw(INT32_MAX); 
        fixed_point b = fixed_point::from_raw(1);
        auto r = a + b;
        assert(r.v == INT32_MAX);
        check_ok("Addition Overflow");
    }

    // ---------- multiplication overflow ----------
    {
        float big_val = 200.0f;
        fixed_point a(big_val);
        fixed_point b(big_val);
        fixed_point r_fixed = a * b; 

        float r_true = (float)big_val * big_val;
        assert(std::abs(r_true - r_fixed.to_float()) > 1.0);

        assert(r_fixed.v == INT32_MAX);
        
        check_ok("Multiplication Overflow");
    }

    // ---------- division overflow (Saturation) ----------
    {
        float val_a = 1000.0;
        float val_b = 0.0001;
        fixed_point a(val_a);
        fixed_point b(val_b);
        auto r = a / b;

        float math_result = val_a / val_b;
        assert(std::abs(math_result - r.to_float()) > 1.0); 
        
        check_ok("Division overflow");
    }

    // ---------- division by zero ----------
    {
        fixed_point a(1.0);
        fixed_point zero;
        auto r = a / zero;
        assert(r.v == INT32_MAX);
        check_ok("Division by zero:", "(saturated to MAX)");
    }

    // ---------- dot accumulation stress ----------
    {
        fixed_point x[8];
        int64_t manual_acc = 0;
        for(int i=0; i<8; i++) {
            x[i] = fixed_point(100.0);
            manual_acc += ((int64_t)x[i].v * x[i].v) >> fixed_point::FRACTION_BITS;
        }
        fixed_point d = dot_fixed(x, x, 8);
        assert(d.v == INT32_MAX);
        check_ok("Dot product overflow");
    }

    // ---------- matrix vector product ----------
    {
        std::cout << "\n=== Fixed-point SpMV demo ===\n";
        const char* testcase_path = nullptr;
        const char* candidates[] = {
            "testcase_preconditioned.bin",
            "tests/testcase_preconditioned.bin",
            "../tests/testcase_preconditioned.bin",
        };
        for (const char* p : candidates) {
            if (std::filesystem::exists(p)) {
                testcase_path = p;
                break;
            }
        }
        assert(testcase_path != nullptr);
        csr_problem_from_file(testcase_path, prob);

        std::cout
            << "Loaded matrix: n="
            << prob.A.n
            << " nnz=" << prob.A.nnz
            << "\n";

        spmv(prob.A, prob.x, y);

        for (size_t i = 0; i < prob.A.n; ++i)
        {
            float yd = y[i].to_float();
            float bd = prob.b[i].to_float();

            float err = std::abs(yd - bd);
            assert(err < EPS);
        }
        check_ok("spmv() test A*x = b");
    
    }

    // ---------- scale + direction ----------
    {
        std::cout << "\n=== Scale + direction tests ===\n";

        fixed_point v[3] = { fixed_point(3.0f), fixed_point(4.0f), fixed_point(0.0f) };
        ScaledDirectionVector<fixed_point> sv{};
        cg_make_scaled_direction(v, 3, sv);

        check_close(5.0f, sv.scale, "Scale-dir norm as scale");
        check_close(0.6f, sv.dir[0].to_float(), "Scale-dir d0");
        check_close(0.8f, sv.dir[1].to_float(), "Scale-dir d1");
        check_close(0.0f, sv.dir[2].to_float(), "Scale-dir d2");

        fixed_point v_rec[3]{};
        cg_reconstruct_from_scaled_direction(sv, v_rec, 3);
        check_close(3.0f, v_rec[0].to_float(), "Scale-dir recon v0");
        check_close(4.0f, v_rec[1].to_float(), "Scale-dir recon v1");
        check_close(0.0f, v_rec[2].to_float(), "Scale-dir recon v2");
    }

    // ---------- scale + direction axpy ----------
    {
        fixed_point x_raw[2] = { fixed_point(1.0f), fixed_point(2.0f) };
        fixed_point d_raw[2] = { fixed_point(0.5f), fixed_point(-1.0f) };

        ScaledDirectionVector<fixed_point> sx{};
        ScaledDirectionVector<fixed_point> sd{};
        ScaledDirectionVector<fixed_point> sout{};
        cg_make_scaled_direction(x_raw, 2, sx);
        cg_make_scaled_direction(d_raw, 2, sd);
        cg_scaled_direction_axpy(sx, 2.0f, sd, sout, 2);

        fixed_point out_raw[2]{};
        cg_reconstruct_from_scaled_direction(sout, out_raw, 2);
        check_close(2.0f, out_raw[0].to_float(), "Scale-dir AXPY v0");
        check_close(0.0f, out_raw[1].to_float(), "Scale-dir AXPY v1");
    }

    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}

// CPU-emulation driver for the cuda-ode kernels of one lane, implicit
// theta-method (THETA=1.0: backward Euler, 0.5: trapezoidal with backward-Euler
// fallback) with Newton + line search and step subdivision down to
// h / 2^MAXDEPTH at saturation breakpoints.
//
// Build (M is the model identifier, U the same in upper case):
//   g++ -std=c++17 -O1 -w -DTHETA=1.0 -DMAXDEPTH=12 \
//     -DMODEL_SRC='"IEEE39Bus_cuda/<M>_ode.cu"' -DM=<M> \
//     -DY_LEN=<U>_Y_LEN -DP_LEN=<U>_P_LEN -DNSTATE=<U>_STATE_COUNT \
//     -DHSTRIDE=<U>_DELAY_HISTORY_STRIDE be_harness.cpp -o be_harness
// Run: ./be_harness <t_end> <h> [states.csv]
// Trapezoidal-rule (implicit) driver for the generated per-lane kernels, one lane.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <functional>
struct Dim3 { int x; };
static Dim3 blockIdx, blockDim, threadIdx;
#define __global__
#define __host__
#define __device__
static void rmc_debug_block(int b, int s, double t) { static int n = 0; if (n++ < 40) printf("block %d failed code %d at t=%.9g\n", b, s, t); }
#include MODEL_SRC
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define K(n) CAT(M, n)
static const int NY = Y_LEN, NP = P_LEN, NS = NSTATE;
static std::vector<double> p(NP), hist(HSTRIDE);
static long rhs_calls = 0, rhs_failures = 0; static double first_failure = -1, cur_t = 0;
static int rhs(double t, std::vector<double>& y, std::vector<double>& out) {
    int st = 0; ++rhs_calls;
    K(_delay_apply_batch)(t, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
    K(_rhs_batch)(t, y.data(), p.data(), out.data(), &st, NY, NP, NS, 1);
    if (st) { ++rhs_failures; if (first_failure < 0) { first_failure = t; printf("first rhs failure status=%d at t=%.9g\n", st, t); } }
    return st;
}
static bool lu(std::vector<double>& a, std::vector<int>& piv, int n) {
    for (int c = 0; c < n; ++c) {
        int m = c; for (int i = c + 1; i < n; ++i) if (fabs(a[i*n+c]) > fabs(a[m*n+c])) m = i;
        if (a[m*n+c] == 0.0) return false;
        piv[c] = m; if (m != c) for (int j = 0; j < n; ++j) std::swap(a[c*n+j], a[m*n+j]);
        for (int i = c + 1; i < n; ++i) { double f = a[i*n+c] /= a[c*n+c]; if (f != 0.0) for (int j = c + 1; j < n; ++j) a[i*n+j] -= f * a[c*n+j]; }
    }
    return true;
}
static void lu_solve(const std::vector<double>& a, const std::vector<int>& piv, int n, std::vector<double>& b) {
    for (int c = 0; c < n; ++c) { std::swap(b[c], b[piv[c]]); for (int i = c + 1; i < n; ++i) b[i] -= a[i*n+c] * b[c]; }
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= a[i*n+j] * b[j]; b[i] = s / a[i*n+i]; }
}
int main(int argc, char** argv) {
    double t_end = argc > 1 ? atof(argv[1]) : 0.01, h = argc > 2 ? atof(argv[2]) : 5e-6;
    const char* csv = argc > 3 ? argv[3] : nullptr;
    std::vector<double> y(NY), f0(NS), f1(NS), fp(NS), r(NS), J(NS*NS);
    std::vector<int> piv(NS);
    int st = 0;
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    K(_load_starts_batch)(y.data(), p.data(), NY, NP, 1);
    K(_initialize_batch)(0.0, y.data(), p.data(), &st, NY, NP, 1);
    if (st) printf("init status %d\n", st);
    K(_delay_init_batch)(0.0, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
    FILE* out = csv ? fopen(csv, "w") : nullptr;
    auto start = std::chrono::steady_clock::now();
    double t = 0; int steps = (int)(t_end / h + 0.5), jac_age = 1 << 30, jacobians = 0, failed = 0, halvings = 0;
    double jac_h = -1;
    rhs(t, y, f0);
    double theta = 0.5, jac_theta = -1; long be_steps = 0;
    auto residual = [&](const std::vector<double>& x, const std::vector<double>& f1, double hh, std::vector<double>& out) {
        double n2 = 0; for (int j = 0; j < NS; ++j) { out[j] = -(x[j] - y[j] - hh * ((1.0 - theta) * f0[j] + theta * f1[j])); n2 += out[j] * out[j]; } return n2; };
    // One trapezoidal step of size hh from (t, y); true when Newton converged.
    double best_err = 0; std::vector<double> best_x; long relaxed = 0;
    auto attempt = [&](double hh, std::vector<double>& x) -> bool {
        best_err = INFINITY;
        double t1 = t + hh;
        x = y; for (int j = 0; j < NS; ++j) x[j] = y[j] + hh * f0[j];
        for (int refresh = 0; refresh < 4; ++refresh) {
            if (refresh > 0 || jac_age > 50 || jac_h != hh || jac_theta != theta) {
                // Jacobian at the start point first, then at the current
                // iterate (a saturation breakpoint changes the slopes).
                const std::vector<double> at = refresh == 0 ? y : x;
                std::vector<double> xb = at; rhs(t1, xb, f1);
                for (int k = 0; k < NS; ++k) {
                    std::vector<double> xp = at; double d = 1e-7 * fmax(1.0, fabs(at[k])); xp[k] += d;
                    rhs(t1, xp, fp);
                    for (int i = 0; i < NS; ++i) J[i*NS+k] = (i == k ? 1.0 : 0.0) - theta * hh * (fp[i] - f1[i]) / d;
                }
                if (!lu(J, piv, NS)) return false;
                jac_age = 0; jac_h = hh; jac_theta = theta; ++jacobians;
            }
            std::vector<double> xt(NY);
            rhs(t1, x, f1); double n2 = residual(x, f1, hh, r);
            for (int it = 0; it < 10; ++it) {
                std::vector<double> dx = r; lu_solve(J, piv, NS, dx);
                double lambda = 1.0, n2new = 0;
                for (int trial = 0; trial < 8; ++trial) {
                    xt = x; for (int j = 0; j < NS; ++j) xt[j] = x[j] + lambda * dx[j];
                    int st = rhs(t1, xt, f1);
                    std::vector<double> rr(NS); n2new = residual(xt, f1, hh, rr);
                    if (!st && std::isfinite(n2new) && (n2new < n2 || trial == 7)) { r = rr; break; }
                    lambda *= 0.5;
                }
                if (!std::isfinite(n2new)) break;
                double norm = 0; for (int j = 0; j < NS; ++j) norm = fmax(norm, fabs(lambda * dx[j]) / (1e-6 + 1e-5 * fabs(xt[j])));
                x = xt; n2 = n2new;
                { double e = 0; for (int j = 0; j < NS; ++j) e = fmax(e, fabs(r[j]) / (1.0 + fabs(x[j]))); if (e < best_err) { best_err = e; best_x = x; } }
                if (norm < 1.0 && lambda == 1.0) return true;
            }
        }
        return false;
    };
    std::function<bool(double, int)> advance = [&](double hh, int depth) -> bool {
        std::vector<double> x;
        theta = THETA;
        bool done = attempt(hh, x);
        if (!done && THETA < 1.0) {
            // Critical damping adjustment: backward Euler (L-stable) over a
            // step the trapezoidal rule cannot take (a saturation breakpoint).
            theta = 1.0; done = attempt(hh, x); be_steps += done;
        }
        if (!done) {
            if (depth >= MAXDEPTH && best_err < 1e-4) { x = best_x; ++relaxed; }
            else if (depth >= MAXDEPTH) { x = y; ++failed; if (failed == 1) printf("first unconverged step at t=%.9g\n", t + hh); return false; }
            else { ++halvings;
            return advance(0.5 * hh, depth + 1) && advance(0.5 * hh, depth + 1); }
        }
        y = x; t += hh; ++jac_age;
        K(_delay_apply_batch)(t, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
        int passes = 0;
        K(_event_update_batch)(t, y.data(), p.data(), &passes, NY, NP, 1);
        K(_project_manifold_batch)(t, y.data(), p.data(), nullptr, NY, NP, 1);
        rhs(t, y, f0);
        K(_delay_commit_batch)(t, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
        return true;
    };
    for (int n = 0; n < steps; ++n) {
        if (!advance(h, 0)) break;
        if (out && (n + 1) % 10 == 0) { fprintf(out, "%.9g", t); for (int j = 0; j < NS; ++j) fprintf(out, ",%.9g", y[j]); fprintf(out, "\n"); }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("rhs_failures=%ld\n", rhs_failures); printf("halvings=%d relaxed=%ld be_steps=%ld\n", halvings, relaxed, be_steps); printf("h=%g t=%g steps=%d unconverged=%d jacobians=%d rhs_calls=%ld wall=%.2fs\n", h, t, steps, failed, jacobians, rhs_calls, secs);
    printf("  "); for (int j = 0; j < 12; ++j) printf(" y[%d]=%.8g", j, y[j]); printf("\n");
    if (out) fclose(out);
}

// gpu_btf.cu: the be_harness_btf.cpp integrator on the GPU, one lane per
// thread, the whole time loop on the device.
//
// Same method as be_harness_btf.cpp with its defaults (BTF=1, REUSE_JF=0,
// BLOCK_NEWTON=0, Newton tolerances 1e-9/1e-8, step subdivision down to
// h / 2^MAXDEPTH). The sparsity pattern comes from one dense finite-difference
// probe of lane 0 at t=0 (no periodic probes: on the 39-bus the pattern never
// grew over 0.1 s). The host builds the coloring and the block triangular form
// once and uploads them; every lane then runs:
//   colored finite-difference Jacobian, per-block LU, block-by-block solve,
//   Newton with line search, backward-Euler fallback when THETA < 1, step
//   subdivision with an explicit stack instead of recursion.
// Each lane owns its y, p, delay history and work arrays in global memory.
// The model kernels are called per lane from device code: while the model
// source is included, __global__ becomes __device__ and the thread indices
// read 0, so every model kernel works on the lane pointers it is given.
//
// The same file builds on the CPU (-DCPU_EMU, one lane after another), which
// reproduces be_harness_btf.cpp built with -DPROBE_EVERY=0 -DMAX_FAIL_PROBES=0
// bit for bit.
//
// GPU (CUDA 12/13; -O3 makes ptxas run out of memory on the 39-bus). The
// model is compiled twice: for the device, and for the host (-DHOST_TU) to
// initialize and probe once on the CPU.
//   g++ -O1 -std=c++17 -w -x c++ -DHOST_TU -DMODEL_SRC='"<M>_ode.cu"'
//     -DM=<M> -DU=<U> -c gpu_btf.cu -o gpu_btf_host.o
//   objcopy --keep-global-symbol=host_init --keep-global-symbol=host_rhs gpu_btf_host.o
//     (the model's globals are defined in both objects; keep only these two)
//   nvcc -O1 -Xcicc -O1 -Xptxas -O1 -std=c++17 -arch=native
//     -DMODEL_SRC='"<M>_ode.cu"' -DM=<M> -DU=<U> gpu_btf.cu gpu_btf_host.o -o gpu_btf
// CPU emulation:
//   g++ -O1 -std=c++17 -w -x c++ -DCPU_EMU -DMODEL_SRC='"<M>_ode.cu"'
//     -DM=<M> -DU=<U> gpu_btf.cu -o gpu_btf_cpu
// Run: ./gpu_btf <t_end> <h> <lanes> [csv_prefix] [out_lanes]
//   writes <csv_prefix>_lane<k>.csv (t, all states every OUT_EVERY steps) for
//   the first out_lanes lanes (default 1).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#ifndef THETA
#define THETA 1.0
#endif
#ifndef MAXDEPTH
#define MAXDEPTH 12
#endif
#ifndef NEWTON_ATOL
#define NEWTON_ATOL 1e-9
#endif
#ifndef NEWTON_RTOL
#define NEWTON_RTOL 1e-8
#endif
#ifndef OUT_EVERY
#define OUT_EVERY 10
#endif
#ifndef CHUNK_STEPS
#define CHUNK_STEPS 400
#endif
#ifndef BLOCK_THREADS
#define BLOCK_THREADS 32
#endif

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define K(n) CAT(M, n)
#define C(n) CAT(U, n)

#if defined(CPU_EMU) || defined(HOST_TU)
struct Dim3 { int x; };
static Dim3 blockIdx{0}, blockDim{1}, threadIdx{0};
using std::isfinite;
#define __global__
#define __host__
#define __device__
static void rmc_debug_block(int, int, double) {}
#include MODEL_SRC
// Host side of the model: initialization and the t=0 probe run here, once.
// On the GPU the inlined initialization needs about 1.3 MB of stack per
// thread, more than a launch accepts. In the CUDA build this part is compiled
// by the host compiler from the same file with -DHOST_TU and linked in.
extern "C" void host_init(double *y, double *p, double *hist, int *status) {
    const int hs = C(_DELAY_HISTORY_STRIDE) > 0 ? C(_DELAY_HISTORY_STRIDE) : 1;
    *status = 0;
    K(_load_starts_batch)(y, p, C(_Y_LEN), C(_P_LEN), 1);
    K(_initialize_batch)(0.0, y, p, status, C(_Y_LEN), C(_P_LEN), 1);
    K(_delay_init_batch)(0.0, y, p, hist, C(_Y_LEN), C(_P_LEN), hs, 1);
}
extern "C" int host_rhs(double t, double *y, double *p, double *hist, double *out) {
    const int hs = C(_DELAY_HISTORY_STRIDE) > 0 ? C(_DELAY_HISTORY_STRIDE) : 1;
    int status = 0;
    K(_delay_apply_batch)(t, y, p, hist, C(_Y_LEN), C(_P_LEN), hs, 1);
    K(_rhs_batch)(t, y, p, out, &status, C(_Y_LEN), C(_P_LEN), C(_STATE_COUNT), 1);
    return status;
}
#endif

#ifndef HOST_TU
#ifdef CPU_EMU
#define HD
#define LAUNCH(kernel, lanes, ...) \
    for (int rmc_l = 0; rmc_l < (lanes); ++rmc_l) kernel(rmc_l, __VA_ARGS__)
#define LANE_ARG int lane,
#define LANE_ID lane
#define CHECK(call) (call)
#else
#include <cuda_runtime.h>
extern "C" void host_init(double *y, double *p, double *hist, int *status);
extern "C" int host_rhs(double t, double *y, double *p, double *hist, double *out);
struct RmcIdx { unsigned int x, y, z; };
static __device__ constexpr RmcIdx rmc_idx_zero{0u, 0u, 0u};
static __device__ constexpr RmcIdx rmc_idx_one{1u, 1u, 1u};
#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#define blockIdx rmc_idx_zero
#define threadIdx rmc_idx_zero
#define blockDim rmc_idx_one
#include MODEL_SRC
#undef blockIdx
#undef threadIdx
#undef blockDim
#pragma pop_macro("__global__")
#define HD __device__
#define LAUNCH(kernel, lanes, ...) \
    kernel<<<((lanes) + BLOCK_THREADS - 1) / BLOCK_THREADS, BLOCK_THREADS>>>(__VA_ARGS__)
#define LANE_ARG
#define LANE_ID (int)(blockIdx.x * blockDim.x + threadIdx.x)
#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call,       \
                    cudaGetErrorString(e));                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#endif

static const int NY = C(_Y_LEN), NP = C(_P_LEN), NS = C(_STATE_COUNT);
static const int HS = C(_DELAY_HISTORY_STRIDE) > 0 ? C(_DELAY_HISTORY_STRIDE) : 1;

// ---- Structure shared by every lane (read-only on the device) -------------
struct Structure {
    int nnz, ncolors, nblocks, lu_len, map_len, noff;
    const int *col_ptr, *ent_row;           // CSC pattern of J (entries by column)
    const int *color_ptr, *color_cols;      // columns of each color
    const int *blk_ptr, *blk_idx;           // states of each block, solve order
    const int *blk_lu, *blk_map;            // LU offset per block; map offset per block
    const int *map;                         // map[blk_map[b] + r*s + c] = CSC entry or -1
    const int *off_ptr, *off_k, *off_j, *off_e;  // off-block entries per block
};

// ---- Per-lane memory -------------------------------------------------------
struct LaneState {
    double t, fact_h, fact_theta, theta, best_err;
    int jac_age, jf_valid, failed, steps;
    long long rhs_calls, rhs_failures, jac_rhs_calls;
    int jacobians, factorizations, halvings, relaxed, be_steps, unconverged;
};
struct Work {
    // offsets (in doubles) into one lane's arena
    int y, x, xt, xp, best, f0, f1, fp, r, rr, dx, d, jf, lu, len;
};
static Work make_work(int nnz, int lu_len) {
    Work w{}; int o = 0;
    w.y = o; o += NY; w.x = o; o += NY; w.xt = o; o += NY; w.xp = o; o += NY; w.best = o; o += NY;
    w.f0 = o; o += NS; w.f1 = o; o += NS; w.fp = o; o += NS; w.r = o; o += NS; w.rr = o; o += NS;
    w.dx = o; o += NS; w.d = o; o += NS; w.jf = o; o += nnz; w.lu = o; o += lu_len; w.len = o;
    return w;
}
struct Lane {
    double *a;        // arena
    double *p, *hist;
    int *piv;
    LaneState *s;
    const Structure *st;
    Work w;
};

HD static int lane_rhs(Lane &L, double t, double *y, double *out) {
    int status = 0; ++L.s->rhs_calls;
    K(_delay_apply_batch)(t, y, L.p, L.hist, NY, NP, HS, 1);
    K(_rhs_batch)(t, y, L.p, out, &status, NY, NP, NS, 1);
    if (status) ++L.s->rhs_failures;
    return status;
}
HD static void copy(double *dst, const double *src, int n) { for (int i = 0; i < n; ++i) dst[i] = src[i]; }

HD static void jacobian(Lane &L, double t1, const double *at) {
    const Structure &S = *L.st; double *A = L.a;
    long long c0 = L.s->rhs_calls;
    double *xp = A + L.w.xp, *f1 = A + L.w.f1, *fp = A + L.w.fp, *d = A + L.w.d, *jf = A + L.w.jf;
    copy(xp, at, NY); lane_rhs(L, t1, xp, f1);
    for (int c = 0; c < S.ncolors; ++c) {
        copy(xp, at, NY);
        for (int q = S.color_ptr[c]; q < S.color_ptr[c + 1]; ++q) {
            int j = S.color_cols[q]; d[j] = 1e-7 * fmax(1.0, fabs(at[j])); xp[j] += d[j];
        }
        lane_rhs(L, t1, xp, fp);
        for (int q = S.color_ptr[c]; q < S.color_ptr[c + 1]; ++q) {
            int j = S.color_cols[q];
            for (int e = S.col_ptr[j]; e < S.col_ptr[j + 1]; ++e) jf[e] = (fp[S.ent_row[e]] - f1[S.ent_row[e]]) / d[j];
        }
    }
    L.s->jac_rhs_calls += L.s->rhs_calls - c0;
}

HD static bool lu(double *a, int *piv, int n) {
    for (int c = 0; c < n; ++c) {
        int m = c; for (int i = c + 1; i < n; ++i) if (fabs(a[i*n+c]) > fabs(a[m*n+c])) m = i;
        if (a[m*n+c] == 0.0) return false;
        piv[c] = m; if (m != c) for (int j = 0; j < n; ++j) { double s = a[c*n+j]; a[c*n+j] = a[m*n+j]; a[m*n+j] = s; }
        for (int i = c + 1; i < n; ++i) { double f = a[i*n+c] /= a[c*n+c]; if (f != 0.0) for (int j = c + 1; j < n; ++j) a[i*n+j] -= f * a[c*n+j]; }
    }
    return true;
}
HD static void lu_solve(const double *a, const int *piv, int n, double *b) {
    for (int c = 0; c < n; ++c) { double s = b[c]; b[c] = b[piv[c]]; b[piv[c]] = s; for (int i = c + 1; i < n; ++i) b[i] -= a[i*n+c] * b[c]; }
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= a[i*n+j] * b[j]; b[i] = s / a[i*n+i]; }
}

HD static bool factor(Lane &L, double theta, double hh) {
    const Structure &S = *L.st; double *jf = L.a + L.w.jf;
    for (int b = 0; b < S.nblocks; ++b) {
        int base = S.blk_ptr[b], s = S.blk_ptr[b + 1] - base;
        double *m = L.a + L.w.lu + S.blk_lu[b]; const int *map = S.map + S.blk_map[b];
        for (int r = 0; r < s; ++r) for (int c = 0; c < s; ++c) {
            int e = map[r*s+c];
            m[r*s+c] = (r == c ? 1.0 : 0.0) - theta * hh * (e >= 0 ? jf[e] : 0.0);
        }
        if (!lu(m, L.piv + base, s)) return false;
    }
    return true;
}

// rb is a scratch of NS doubles (the fp slot is free during a solve).
HD static void solve(Lane &L, double theta, double hh, double *dx) {
    const Structure &S = *L.st; double *jf = L.a + L.w.jf, *rb = L.a + L.w.fp;
    for (int b = 0; b < S.nblocks; ++b) {
        int base = S.blk_ptr[b], s = S.blk_ptr[b + 1] - base;
        for (int k = 0; k < s; ++k) rb[k] = dx[S.blk_idx[base + k]];
        for (int q = S.off_ptr[b]; q < S.off_ptr[b + 1]; ++q) rb[S.off_k[q]] += theta * hh * jf[S.off_e[q]] * dx[S.off_j[q]];
        lu_solve(L.a + L.w.lu + S.blk_lu[b], L.piv + base, s, rb);
        for (int k = 0; k < s; ++k) dx[S.blk_idx[base + k]] = rb[k];
    }
}

HD static double residual(Lane &L, const double *x, const double *f1, double hh, double *out) {
    const double *y = L.a + L.w.y, *f0 = L.a + L.w.f0; double theta = L.s->theta, n2 = 0;
    for (int j = 0; j < NS; ++j) { out[j] = -(x[j] - y[j] - hh * ((1.0 - theta) * f0[j] + theta * f1[j])); n2 += out[j] * out[j]; }
    return n2;
}

// One implicit step of size hh from (t, y) into x; true when Newton converged.
HD static bool attempt(Lane &L, double hh) {
    LaneState &s = *L.s; double *A = L.a;
    double *y = A + L.w.y, *x = A + L.w.x, *xt = A + L.w.xt, *f0 = A + L.w.f0, *f1 = A + L.w.f1;
    double *r = A + L.w.r, *rr = A + L.w.rr, *dx = A + L.w.dx, *best = A + L.w.best;
    s.best_err = INFINITY;
    double t1 = s.t + hh;
    copy(x, y, NY); for (int j = 0; j < NS; ++j) x[j] = y[j] + hh * f0[j];
    for (int refresh = 0; refresh < 4; ++refresh) {
        bool new_jf = refresh > 0 || s.jac_age > 50 || !s.jf_valid || s.fact_h != hh || s.fact_theta != s.theta;
        if (new_jf) {
            // Jacobian at the start point first, then at the current iterate.
            // xt is free here: keep a copy of the point, as the RHS settles
            // the algebraics in place.
            copy(xt, refresh == 0 ? y : x, NY);
            jacobian(L, t1, xt);
            s.jac_age = 0; s.jf_valid = 1; ++s.jacobians;
        }
        if (new_jf || s.fact_h != hh || s.fact_theta != s.theta) {
            if (!factor(L, s.theta, hh)) { s.fact_h = -1; return false; }
            s.fact_h = hh; s.fact_theta = s.theta; ++s.factorizations;
        }
        lane_rhs(L, t1, x, f1); double n2 = residual(L, x, f1, hh, r);
        for (int it = 0; it < 10; ++it) {
            copy(dx, r, NS); solve(L, s.theta, hh, dx);
            double lambda = 1.0, n2new = 0;
            for (int trial = 0; trial < 8; ++trial) {
                copy(xt, x, NY); for (int j = 0; j < NS; ++j) xt[j] = x[j] + lambda * dx[j];
                int st = lane_rhs(L, t1, xt, f1);
                n2new = residual(L, xt, f1, hh, rr);
                if (!st && isfinite(n2new) && (n2new < n2 || trial == 7)) { copy(r, rr, NS); break; }
                lambda *= 0.5;
            }
            if (!isfinite(n2new)) break;
            double norm = 0; for (int j = 0; j < NS; ++j) norm = fmax(norm, fabs(lambda * dx[j]) / (NEWTON_ATOL + NEWTON_RTOL * fabs(xt[j])));
            copy(x, xt, NY); n2 = n2new;
            { double e = 0; for (int j = 0; j < NS; ++j) e = fmax(e, fabs(r[j]) / (1.0 + fabs(x[j]))); if (e < s.best_err) { s.best_err = e; copy(best, x, NY); } }
            if (norm < 1.0 && lambda == 1.0) return true;
        }
    }
    return false;
}

HD static void accept(Lane &L, double hh, const double *x) {
    LaneState &s = *L.s; double *y = L.a + L.w.y;
    copy(y, x, NY); s.t += hh; ++s.jac_age;
    K(_delay_apply_batch)(s.t, y, L.p, L.hist, NY, NP, HS, 1);
    int passes = 0;
    K(_event_update_batch)(s.t, y, L.p, &passes, NY, NP, 1);
    K(_project_manifold_batch)(s.t, y, L.p, nullptr, NY, NP, 1);
    lane_rhs(L, s.t, y, L.a + L.w.f0);
    K(_delay_commit_batch)(s.t, y, L.p, L.hist, NY, NP, HS, 1);
}

// One top-level step of size h, subdividing on failure (depth-first, in time
// order, as the recursion of be_harness_btf.cpp). False when a lane failed.
HD static bool advance(Lane &L, double h) {
    LaneState &s = *L.s;
    int stack[MAXDEPTH + 2]; int top = 0; stack[top++] = 0;
    while (top > 0) {
        int depth = stack[--top];
        double hh = h; for (int k = 0; k < depth; ++k) hh *= 0.5;
        s.theta = THETA;
        bool done = attempt(L, hh);
        if (!done && THETA < 1.0) { s.theta = 1.0; done = attempt(L, hh); s.be_steps += done; }
        const double *x = L.a + L.w.x;
        if (!done) {
            if (depth >= MAXDEPTH && s.best_err < 1e-4) { x = L.a + L.w.best; ++s.relaxed; }
            else if (depth >= MAXDEPTH) { ++s.unconverged; s.failed = 1; return false; }
            else { ++s.halvings; stack[top++] = depth + 1; stack[top++] = depth + 1; continue; }
        }
        accept(L, hh, x);
    }
    return true;
}

struct Buffers {
    double *arena, *p, *hist, *out; int *piv; LaneState *state; Structure *st;
    Work w; int lanes, out_lanes, out_rows;
};

// One RHS of lane 0 at its current y and history, on the device, into fp
// (y copied to xp first). Used to compare device and host model arithmetic.
#ifdef CPU_EMU
static void k_rhs_check(int lane, Buffers B, double t) {
#else
__global__ void k_rhs_check(Buffers B, double t) {
    int lane = LANE_ID;
#endif
    if (lane != 0) return;
    Lane L{B.arena, B.p, B.hist, B.piv, B.state, B.st, B.w};
    long long calls = L.s->rhs_calls, fails = L.s->rhs_failures;
    copy(L.a + L.w.xp, L.a + L.w.y, NY);
    lane_rhs(L, t, L.a + L.w.xp, L.a + L.w.fp);
    L.s->rhs_calls = calls; L.s->rhs_failures = fails;
}

// First RHS (f0 at t=0), then `steps` top-level steps, recording lanes below
// out_lanes every OUT_EVERY steps.
#ifdef CPU_EMU
static void k_run(int lane, Buffers B, int first, int steps, double h, int start) {
#else
__global__ void k_run(Buffers B, int first, int steps, double h, int start) {
    int lane = LANE_ID;
#endif
    if (lane >= B.lanes) return;
    Lane L{B.arena + (size_t)lane * B.w.len, B.p + (size_t)lane * NP, B.hist + (size_t)lane * HS,
           B.piv + (size_t)lane * NS, B.state + lane, B.st, B.w};
    LaneState &s = *L.s;
    if (first) lane_rhs(L, s.t, L.a + L.w.y, L.a + L.w.f0);
    for (int n = 0; n < steps && !s.failed; ++n) {
        if (!advance(L, h)) break;
        ++s.steps;
        int global = start + n + 1;
        if (lane < B.out_lanes && global % OUT_EVERY == 0) {
            int row = global / OUT_EVERY - 1;
            if (row < B.out_rows) {
                double *o = B.out + ((size_t)lane * B.out_rows + row) * (NS + 1);
                o[0] = s.t; copy(o + 1, L.a + L.w.y, NS);
            }
        }
    }
}

// ---- Host: structure from the probed pattern ------------------------------
struct HostStructure {
    std::vector<int> col_ptr, ent_row, color_ptr, color_cols, blk_ptr, blk_idx, blk_lu, blk_map, map, off_ptr, off_k, off_j, off_e;
    int ncolors = 0, nblocks = 0, lu_len = 0, largest = 0;
};
static HostStructure build(const std::vector<unsigned char> &P) {
    HostStructure H;
    std::vector<std::vector<int>> col_rows(NS), row_cols(NS);
    for (int i = 0; i < NS; ++i) for (int j = 0; j < NS; ++j) if (P[i*NS+j]) { col_rows[j].push_back(i); row_cols[i].push_back(j); }
    H.col_ptr.assign(NS + 1, 0);
    for (int j = 0; j < NS; ++j) { H.col_ptr[j + 1] = H.col_ptr[j] + (int)col_rows[j].size(); for (int i : col_rows[j]) H.ent_row.push_back(i); }
    auto entry = [&](int i, int j) { for (int e = H.col_ptr[j]; e < H.col_ptr[j + 1]; ++e) if (H.ent_row[e] == i) return e; return -1; };
    // Coloring, as be_harness_btf.cpp.
    std::vector<std::vector<int>> conflict(NS);
    { std::vector<int> mark(NS, -1);
      for (int j = 0; j < NS; ++j) { mark[j] = j; for (int i : col_rows[j]) for (int k : row_cols[i]) if (mark[k] != j) { mark[k] = j; conflict[j].push_back(k); } } }
    std::vector<int> order(NS); for (int j = 0; j < NS; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return conflict[a].size() > conflict[b].size(); });
    std::vector<int> color(NS, -1);
    for (int j : order) {
        std::vector<char> used(H.ncolors + 1, 0);
        for (int k : conflict[j]) if (color[k] >= 0) used[color[k]] = 1;
        int c = 0; while (used[c]) ++c;
        color[j] = c; H.ncolors = std::max(H.ncolors, c + 1);
    }
    H.color_ptr.assign(H.ncolors + 1, 0);
    for (int c = 0; c < H.ncolors; ++c) { for (int j = 0; j < NS; ++j) if (color[j] == c) H.color_cols.push_back(j); H.color_ptr[c + 1] = (int)H.color_cols.size(); }
    // Tarjan on i -> j, as be_harness_btf.cpp.
    std::vector<int> index(NS, -1), low(NS, 0), stack; std::vector<char> on(NS, 0);
    std::vector<std::vector<int>> comps; int counter = 0;
    for (int s0 = 0; s0 < NS; ++s0) {
        if (index[s0] >= 0) continue;
        std::vector<std::pair<int, size_t>> work{{s0, 0}};
        while (!work.empty()) {
            auto &[v, k] = work.back();
            if (k == 0 && index[v] < 0) { index[v] = low[v] = counter++; stack.push_back(v); on[v] = 1; }
            bool descended = false;
            while (k < row_cols[v].size()) {
                int w = row_cols[v][k++];
                if (index[w] < 0) { work.push_back({w, 0}); descended = true; break; }
                if (on[w]) low[v] = std::min(low[v], index[w]);
            }
            if (descended) continue;
            int vv = v; work.pop_back();
            if (low[vv] == index[vv]) {
                std::vector<int> comp;
                while (true) { int w = stack.back(); stack.pop_back(); on[w] = 0; comp.push_back(w); if (w == vv) break; }
                std::sort(comp.begin(), comp.end()); comps.push_back(comp);
            }
            if (!work.empty()) { int u = work.back().first; low[u] = std::min(low[u], low[vv]); }
        }
    }
    std::vector<int> blk_of(NS);
    H.nblocks = (int)comps.size(); H.blk_ptr.assign(H.nblocks + 1, 0); H.off_ptr.assign(H.nblocks + 1, 0);
    for (int b = 0; b < H.nblocks; ++b) for (int i : comps[b]) blk_of[i] = b;
    int map_len = 0;
    for (int b = 0; b < H.nblocks; ++b) {
        auto &cb = comps[b]; int s = (int)cb.size(); H.largest = std::max(H.largest, s);
        for (int i : cb) H.blk_idx.push_back(i);
        H.blk_ptr[b + 1] = (int)H.blk_idx.size();
        H.blk_lu.push_back(H.lu_len); H.lu_len += s * s;
        H.blk_map.push_back(map_len); map_len += s * s;
        for (int r = 0; r < s; ++r) for (int c = 0; c < s; ++c) H.map.push_back(P[cb[r]*NS+cb[c]] ? entry(cb[r], cb[c]) : -1);
        for (int k = 0; k < s; ++k) for (int j : row_cols[cb[k]]) if (blk_of[j] != b) {
            H.off_k.push_back(k); H.off_j.push_back(j); H.off_e.push_back(entry(cb[k], j));
        }
        H.off_ptr[b + 1] = (int)H.off_k.size();
    }
    return H;
}

template <class T> static T *upload(const std::vector<T> &v) {
    if (v.empty()) return nullptr;
#ifdef CPU_EMU
    T *d = new T[v.size()]; std::copy(v.begin(), v.end(), d); return d;
#else
    T *d = nullptr; CHECK(cudaMalloc(&d, sizeof(T) * v.size()));
    CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice)); return d;
#endif
}
template <class T> static T *alloc(size_t n) {
#ifdef CPU_EMU
    return new T[n]();
#else
    T *d = nullptr; CHECK(cudaMalloc(&d, sizeof(T) * n)); CHECK(cudaMemset(d, 0, sizeof(T) * n)); return d;
#endif
}
template <class T> static void upload_to(T *dst, const std::vector<T> &v) {
#ifdef CPU_EMU
    std::copy(v.begin(), v.end(), dst);
#else
    CHECK(cudaMemcpy(dst, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
#endif
}
template <class T> static void download(std::vector<T> &dst, const T *src, size_t n) {
    dst.resize(n);
#ifdef CPU_EMU
    std::copy(src, src + n, dst.begin());
#else
    CHECK(cudaMemcpy(dst.data(), src, sizeof(T) * n, cudaMemcpyDeviceToHost));
#endif
}
static void sync();
// Device RHS against host RHS at lane 0's current state.
static void rhs_check(const Buffers &B, double t, const char *when) {
    LAUNCH(k_rhs_check, 1, B, t); sync();
    std::vector<double> a, hp, hh; download(a, B.arena, (size_t)B.w.len);
    download(hp, B.p, (size_t)NP + 1); download(hh, B.hist, (size_t)HS);
    std::vector<double> y(a.begin() + B.w.y, a.begin() + B.w.y + NY), f(NS);
    int st = host_rhs(t, y.data(), hp.data(), hh.data(), f.data());
    double worst = 0, scale = 0; int at = -1, nd = 0;
    for (int i = 0; i < NS; ++i) {
        double g = a[B.w.fp + i], d = fabs(g - f[i]), r = d / fmax(fabs(f[i]), 1e-300);
        scale = fmax(scale, fabs(f[i])); nd += g != f[i];
        if (r > worst) { worst = r; at = i; }
    }
    printf("rhs check %s (t=%g, host status %d): %d of %d derivatives differ, worst relative %.3g at %d (host %.17g device %.17g), max |f|=%.3g\n",
           when, t, st, nd, NS, worst, at, at >= 0 ? f[at] : 0.0, at >= 0 ? a[B.w.fp + at] : 0.0, scale);
}
static void sync() {
#ifndef CPU_EMU
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
#endif
}

int main(int argc, char **argv) {
    double t_end = argc > 1 ? atof(argv[1]) : 0.01, h = argc > 2 ? atof(argv[2]) : 6.25e-6;
    int lanes = argc > 3 ? atoi(argv[3]) : 1;
    const char *prefix = argc > 4 ? argv[4] : nullptr;
    int out_lanes = prefix ? (argc > 5 ? atoi(argv[5]) : 1) : 0;
    out_lanes = std::min(out_lanes, lanes);
    int steps = (int)(t_end / h + 0.5), out_rows = steps / OUT_EVERY;
    auto wall0 = std::chrono::steady_clock::now();
#ifndef CPU_EMU
    {
        // Stack per thread: what k_run needs, plus a margin. The driver
        // reserves it for every resident thread, so do not ask for more.
        cudaFuncAttributes fa; CHECK(cudaFuncGetAttributes(&fa, k_run));
        size_t stack = fa.localSizeBytes + 4096;
        CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack));
        printf("k_run: %d registers, %zu bytes local, stack limit %zu\n", fa.numRegs, (size_t)fa.localSizeBytes, stack);
    }
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.name);
#else
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    printf("device: CPU emulation\n");
#endif
    printf("THETA=%g MAXDEPTH=%d ATOL=%g RTOL=%g lanes=%d h=%g t_end=%g steps=%d NY=%d NS=%d\n",
           (double)THETA, MAXDEPTH, (double)NEWTON_ATOL, (double)NEWTON_RTOL, lanes, h, t_end, steps, NY, NS);

    // Pass 1: a provisional work layout (dense pattern sizes) for the probe.
    Buffers B{};
    B.lanes = lanes; B.out_lanes = out_lanes; B.out_rows = out_rows;
    B.w = make_work(NS * NS, 0);
    B.p = alloc<double>((size_t)NP * lanes + 1); B.hist = alloc<double>((size_t)HS * lanes);
    B.piv = alloc<int>((size_t)NS * lanes); B.state = alloc<LaneState>(lanes);
    // Initialization on the host (identical for every lane).
    std::vector<double> y0(NY, 0.0), p0(NP + 1, 0.0), hist0(HS, 0.0);
    int init_status = 0;
    host_init(y0.data(), p0.data(), hist0.data(), &init_status);
    if (init_status) printf("initialize status=%d\n", init_status);
    {
        // Dense finite-difference probe at t=0, same steps as be_harness_btf.
        std::vector<double> xp(NY), f1(NS), fp(NS), hJ((size_t)NS * NS);
        std::copy(y0.begin(), y0.end(), xp.begin());
        host_rhs(0.0, xp.data(), p0.data(), hist0.data(), f1.data());
        for (int k = 0; k < NS; ++k) {
            std::copy(y0.begin(), y0.end(), xp.begin());
            double d = 1e-7 * fmax(1.0, fabs(y0[k])); xp[k] += d;
            host_rhs(0.0, xp.data(), p0.data(), hist0.data(), fp.data());
            for (int i = 0; i < NS; ++i) hJ[(size_t)i*NS+k] = fp[i] - f1[i];
        }
        std::vector<unsigned char> pat(NS * NS, 0); int nnz = 0;
        for (int i = 0; i < NS; ++i) for (int j = 0; j < NS; ++j) { pat[i*NS+j] = (i == j) || hJ[i*NS+j] != 0.0; nnz += pat[i*NS+j]; }
        HostStructure H = build(pat);
        long off = (long)H.off_k.size();
        printf("structure: nnz=%d colors=%d blocks=%d largest=%d off_block_nnz=%ld\n", nnz, H.ncolors, H.nblocks, H.largest, off);
        Structure S{};
        S.nnz = nnz; S.ncolors = H.ncolors; S.nblocks = H.nblocks; S.lu_len = H.lu_len; S.map_len = (int)H.map.size(); S.noff = (int)off;
        S.col_ptr = upload(H.col_ptr); S.ent_row = upload(H.ent_row); S.color_ptr = upload(H.color_ptr); S.color_cols = upload(H.color_cols);
        S.blk_ptr = upload(H.blk_ptr); S.blk_idx = upload(H.blk_idx); S.blk_lu = upload(H.blk_lu); S.blk_map = upload(H.blk_map);
        S.map = upload(H.map); S.off_ptr = upload(H.off_ptr);
        S.off_k = upload(H.off_k); S.off_j = upload(H.off_j); S.off_e = upload(H.off_e);
        B.st = upload(std::vector<Structure>{S});
        B.w = make_work(nnz, H.lu_len);
    }
    // Pass 2: every lane with the sparse layout.
    B.arena = alloc<double>((size_t)B.w.len * lanes);
    B.out = alloc<double>((size_t)std::max(1, out_lanes) * std::max(1, out_rows) * (NS + 1));
    printf("memory per lane: %.1f KiB work + %.1f KiB p + %.1f KiB history\n",
           B.w.len * 8.0 / 1024, NP * 8.0 / 1024, HS * 8.0 / 1024);
    {
        std::vector<double> ha((size_t)B.w.len * lanes, 0.0), hp((size_t)NP * lanes + 1, 0.0), hh((size_t)HS * lanes, 0.0);
        std::vector<LaneState> hs0(lanes);
        LaneState s0{}; s0.t = 0; s0.fact_h = -1; s0.fact_theta = -1; s0.theta = 0.5; s0.jac_age = 1 << 30;
        s0.failed = init_status != 0 ? 2 : 0;
        for (int l = 0; l < lanes; ++l) {
            std::copy(y0.begin(), y0.end(), ha.begin() + (size_t)l * B.w.len + B.w.y);
            std::copy(p0.begin(), p0.begin() + NP, hp.begin() + (size_t)l * NP);
            std::copy(hist0.begin(), hist0.end(), hh.begin() + (size_t)l * HS);
            hs0[l] = s0;
        }
        upload_to(B.arena, ha); upload_to(B.p, hp); upload_to(B.hist, hh); upload_to(B.state, hs0);
    }
    rhs_check(B, 0.0, "at t=0");
    auto run0 = std::chrono::steady_clock::now();
    for (int start = 0; start < steps; start += CHUNK_STEPS) {
        int n = std::min(CHUNK_STEPS, steps - start);
        LAUNCH(k_run, lanes, B, start == 0, n, h, start); sync();
        std::vector<LaneState> s0; download(s0, B.state, 1);
        printf("t=%.6g lane0: halvings=%d rhs_calls=%lld wall=%.1fs\n", s0[0].t, s0[0].halvings, s0[0].rhs_calls,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - run0).count());
        rhs_check(B, s0[0].t, "after chunk");
        fflush(stdout);
    }
    double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - run0).count();

    std::vector<LaneState> hs; download(hs, B.state, lanes);
    int failed = 0; long long rhs_min = -1, rhs_max = 0; double t_min = 1e300;
    for (auto &s : hs) { failed += s.failed != 0; rhs_min = rhs_min < 0 ? s.rhs_calls : std::min(rhs_min, s.rhs_calls); rhs_max = std::max(rhs_max, s.rhs_calls); t_min = std::min(t_min, s.t); }
    const LaneState &s = hs[0];
    printf("lane0: t=%g steps=%d unconverged=%d halvings=%d relaxed=%d be_steps=%d jacobians=%d factorizations=%d rhs_calls=%lld (jacobian %lld) rhs_failures=%lld\n",
           s.t, s.steps, s.unconverged, s.halvings, s.relaxed, s.be_steps, s.jacobians, s.factorizations, s.rhs_calls, s.jac_rhs_calls, s.rhs_failures);
    printf("lanes: %d, failed=%d, min t=%g, rhs_calls min=%lld max=%lld\n", lanes, failed, t_min, rhs_min, rhs_max);
    // All lanes start identical: they must end identical.
    std::vector<double> arena; download(arena, B.arena, (size_t)B.w.len * lanes);
    double spread = 0;
    for (int l = 1; l < lanes; ++l) for (int j = 0; j < NS; ++j)
        spread = fmax(spread, fabs(arena[(size_t)l * B.w.len + B.w.y + j] - arena[B.w.y + j]));
    printf("max |y(lane) - y(lane 0)| at the end: %g\n", spread);
    printf("run wall=%.2fs (%.3f s per lane-simulation), total wall=%.2fs\n", run_s, run_s / lanes,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
    printf("  "); for (int j = 0; j < 12 && j < NS; ++j) printf(" y[%d]=%.8g", j, arena[B.w.y + j]); printf("\n");
    if (prefix && out_lanes > 0) {
        std::vector<double> out; download(out, B.out, (size_t)out_lanes * out_rows * (NS + 1));
        for (int l = 0; l < out_lanes; ++l) {
            char name[512]; snprintf(name, sizeof name, "%s_lane%d.csv", prefix, l);
            FILE *f = fopen(name, "w"); if (!f) { perror(name); continue; }
            for (int row = 0; row < out_rows; ++row) {
                const double *o = out.data() + ((size_t)l * out_rows + row) * (NS + 1);
                if (o[0] == 0.0 && row > 0) break;
                fprintf(f, "%.9g", o[0]); for (int j = 0; j < NS; ++j) fprintf(f, ",%.9g", o[1 + j]); fprintf(f, "\n");
            }
            fclose(f);
        }
    }
    return failed ? 1 : 0;
}
#endif  // HOST_TU

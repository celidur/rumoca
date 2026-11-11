// be_harness_btf.cpp: be_harness.cpp with a sparse Newton matrix.
//
// Same integrator as be_harness.cpp (implicit theta-method, Newton + line
// search, backward-Euler fallback, step subdivision down to h / 2^MAXDEPTH).
// Only the Jacobian and the linear solve change:
//
//   1. Sparsity pattern of J = df/dx, found once by a dense finite-difference
//      probe at t=0 (NS+1 RHS calls), then kept up to date: a dense probe runs
//      every PROBE_EVERY accepted steps and after a failed step at depth 0.
//      The pattern only grows; each growth is printed.
//   2. Column coloring (greedy, largest degree first): one perturbed RHS call
//      per color instead of one per state (about 12 instead of 322 for the
//      39-bus at t=0).
//   3. Block triangular form of M = I - theta*h*J: Tarjan SCCs on the pattern,
//      solved block by block in dependency order, dense LU with partial
//      pivoting per diagonal block, off-diagonal blocks applied as sparse
//      products.
//   4. J (of f) is stored apart from M. With -DREUSE_JF=1 a change of h or
//      theta (step halving, backward-Euler fallback) only refactors M, without
//      new RHS calls. The default (0) keeps the original rule, a new
//      finite-difference Jacobian whenever h or theta changes: on the 39-bus
//      over 0.1 s it was 13 % faster overall (fewer halvings).
//
//   5. -DBLOCK_NEWTON=1: the Newton iteration runs per independent group of
//      blocks (blocks joined by off-block entries form one group; on the
//      39-bus every block is its own group). Each group has its own line
//      search step and its own convergence test, and a converged group stops
//      moving. One RHS call still evaluates every group at once, so a group
//      that needs a short step no longer shortens the step of the others.
//      With a single group it is the original iteration, bit for bit.
//
// Newton step tolerance: |dx_i| < NEWTON_ATOL + NEWTON_RTOL*|x_i| (defaults
// 1e-6 and 1e-5, as in be_harness.cpp).
//
// -DBTF=0 runs the same code with a full pattern (NS colors, one block), which
// is the original dense method with the same instrumentation, for A/B timing.
//
// Build (M is the model identifier, U the same in upper case):
//   g++ -std=c++17 -O1 -w -DTHETA=1.0 -DMAXDEPTH=12 \
//     -DMODEL_SRC='"IEEE39Bus_cuda/<M>_ode.cu"' -DM=<M> \
//     -DY_LEN=<U>_Y_LEN -DP_LEN=<U>_P_LEN -DNSTATE=<U>_STATE_COUNT \
//     -DHSTRIDE=<U>_DELAY_HISTORY_STRIDE be_harness_btf.cpp -o be_harness_btf
// Run: ./be_harness_btf <t_end> <h> [states.csv]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
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
#ifndef BTF
#define BTF 1
#endif
#ifndef REUSE_JF
#define REUSE_JF 0
#endif
#ifndef BLOCK_NEWTON
#define BLOCK_NEWTON 0
#endif
#ifndef NEWTON_ATOL
#define NEWTON_ATOL 1e-6
#endif
#ifndef NEWTON_RTOL
#define NEWTON_RTOL 1e-5
#endif
#ifndef PROBE_EVERY
#define PROBE_EVERY 400
#endif
#ifndef MAX_FAIL_PROBES
#define MAX_FAIL_PROBES 50
#endif
static const int NY = Y_LEN, NP = P_LEN, NS = NSTATE;
static std::vector<double> p(NP), hist(HSTRIDE);
static long rhs_calls = 0, rhs_failures = 0; static double first_failure = -1;
using clk = std::chrono::steady_clock;
static double secs_since(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
static double t_rhs = 0, t_jac = 0, t_lu = 0, t_solve = 0, t_probe = 0;
static long jac_rhs_calls = 0, probe_rhs_calls = 0;
static int rhs(double t, std::vector<double>& y, std::vector<double>& out) {
    auto a = clk::now();
    int st = 0; ++rhs_calls;
    K(_delay_apply_batch)(t, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
    K(_rhs_batch)(t, y.data(), p.data(), out.data(), &st, NY, NP, NS, 1);
    if (st) { ++rhs_failures; if (first_failure < 0) { first_failure = t; printf("first rhs failure status=%d at t=%.9g\n", st, t); } }
    t_rhs += secs_since(a);
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
static void lu_solve(const std::vector<double>& a, const std::vector<int>& piv, int n, double* b) {
    for (int c = 0; c < n; ++c) { std::swap(b[c], b[piv[c]]); for (int i = c + 1; i < n; ++i) b[i] -= a[i*n+c] * b[c]; }
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= a[i*n+j] * b[j]; b[i] = s / a[i*n+i]; }
}

// ---- Sparse structure: pattern, coloring, block triangular form ----------
struct Block {
    std::vector<int> idx;                 // states of the block (global indices)
    std::vector<double> lu; std::vector<int> piv;
    std::vector<int> off_i, off_j;        // off-block entries J[i][j], i in block
};
static std::vector<unsigned char> P(NS * NS, 0);  // P[i*NS+j]: J[i][j] may be nonzero
static std::vector<std::vector<int>> col_rows;    // rows of each column
static std::vector<std::vector<int>> groups;      // columns of each color
static std::vector<Block> blocks;                 // in solve order
static std::vector<std::vector<int>> ngroups;     // states of each independent group
static std::vector<double> Jf(NS * NS, 0.0);      // J = df/dx on the pattern
static long pattern_nnz = 0; static int pattern_builds = 0;

static void build_structure() {
    ++pattern_builds;
    pattern_nnz = 0;
    col_rows.assign(NS, {});
    std::vector<std::vector<int>> row_cols(NS);
    for (int i = 0; i < NS; ++i) for (int j = 0; j < NS; ++j)
        if (P[i*NS+j]) { col_rows[j].push_back(i); row_cols[i].push_back(j); ++pattern_nnz; }
    // Coloring: columns that share a row conflict.
    std::vector<std::vector<int>> conflict(NS);
    {
        std::vector<int> mark(NS, -1);
        for (int j = 0; j < NS; ++j) {
            mark[j] = j;
            for (int i : col_rows[j]) for (int k : row_cols[i]) if (mark[k] != j) { mark[k] = j; conflict[j].push_back(k); }
        }
    }
    std::vector<int> order(NS); for (int j = 0; j < NS; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return conflict[a].size() > conflict[b].size(); });
    std::vector<int> color(NS, -1); int ncolors = 0;
    for (int j : order) {
        std::vector<char> used(ncolors + 1, 0);
        for (int k : conflict[j]) if (color[k] >= 0) used[color[k]] = 1;
        int c = 0; while (used[c]) ++c;
        color[j] = c; ncolors = std::max(ncolors, c + 1);
    }
    groups.assign(ncolors, {});
    for (int j = 0; j < NS; ++j) groups[color[j]].push_back(j);
    // Tarjan on edges i -> j (row i depends on unknown j). SCCs come out
    // dependencies first, which is the forward solve order.
    std::vector<int> index(NS, -1), low(NS, 0), stack; std::vector<char> on(NS, 0);
    std::vector<std::vector<int>> comps; int counter = 0;
    for (int s = 0; s < NS; ++s) {
        if (index[s] >= 0) continue;
        std::vector<std::pair<int, size_t>> work{{s, 0}};
        while (!work.empty()) {
            auto& [v, k] = work.back();
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
                std::sort(comp.begin(), comp.end());
                comps.push_back(comp);
            }
            if (!work.empty()) { int u = work.back().first; low[u] = std::min(low[u], low[vv]); }
        }
    }
    std::vector<int> blk_of(NS);
    blocks.assign(comps.size(), {});
    for (size_t b = 0; b < comps.size(); ++b) { blocks[b].idx = comps[b]; for (int i : comps[b]) blk_of[i] = (int)b; }
    long off = 0; size_t largest = 0;
    for (size_t b = 0; b < blocks.size(); ++b) {
        auto& B = blocks[b]; size_t s = B.idx.size(); largest = std::max(largest, s);
        B.lu.assign(s * s, 0.0); B.piv.assign(s, 0);
        for (int i : B.idx) for (int j : row_cols[i]) if (blk_of[j] != (int)b) { B.off_i.push_back(i); B.off_j.push_back(j); ++off; }
    }
    // Independent groups: blocks joined by off-block entries (union-find).
    std::vector<int> parent(blocks.size());
    for (size_t b = 0; b < blocks.size(); ++b) parent[b] = (int)b;
    std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
    for (auto& B : blocks) for (size_t e = 0; e < B.off_i.size(); ++e) {
        int a = find(blk_of[B.off_i[e]]), c = find(blk_of[B.off_j[e]]);
        if (a != c) parent[a] = c;
    }
    std::vector<int> gid(blocks.size(), -1); ngroups.clear();
    for (size_t b = 0; b < blocks.size(); ++b) {
        int root = find((int)b);
        if (gid[root] < 0) { gid[root] = (int)ngroups.size(); ngroups.push_back({}); }
        for (int i : blocks[b].idx) ngroups[gid[root]].push_back(i);
    }
    size_t glargest = 0; for (auto& g : ngroups) glargest = std::max(glargest, g.size());
    printf("structure #%d: nnz=%ld colors=%d blocks=%zu largest=%zu off_block_nnz=%ld groups=%zu largest_group=%zu\n",
           pattern_builds, pattern_nnz, ncolors, blocks.size(), largest, off, ngroups.size(), glargest);
}

// Colored finite-difference Jacobian of f at `at` (time t1); f1 gets f(at).
static void jacobian(double t1, const std::vector<double>& at, std::vector<double>& f1) {
    auto a = clk::now(); long c0 = rhs_calls;
    std::vector<double> xb = at, fp(NS), xp, d(NS);
    rhs(t1, xb, f1);
    for (auto& g : groups) {
        xp = at;
        for (int j : g) { d[j] = 1e-7 * fmax(1.0, fabs(at[j])); xp[j] += d[j]; }
        rhs(t1, xp, fp);
        for (int j : g) for (int i : col_rows[j]) Jf[i*NS+j] = (fp[i] - f1[i]) / d[j];
    }
    jac_rhs_calls += rhs_calls - c0; t_jac += secs_since(a);
}

// LU of every diagonal block of M = I - theta*hh*J.
static bool factor(double theta, double hh) {
    auto a = clk::now(); bool ok = true;
    for (auto& B : blocks) {
        int s = (int)B.idx.size();
        for (int r = 0; r < s; ++r) for (int c = 0; c < s; ++c) {
            int i = B.idx[r], j = B.idx[c];
            B.lu[r*s+c] = (r == c ? 1.0 : 0.0) - theta * hh * (P[i*NS+j] ? Jf[i*NS+j] : 0.0);
        }
        if (!lu(B.lu, B.piv, s)) { ok = false; break; }
    }
    t_lu += secs_since(a);
    return ok;
}

// Solve M dx = r in place (dx holds r on entry), block by block.
static void solve(double theta, double hh, std::vector<double>& dx) {
    auto a = clk::now(); std::vector<double> rb;
    for (auto& B : blocks) {
        int s = (int)B.idx.size(); rb.resize(s);
        for (int k = 0; k < s; ++k) rb[k] = dx[B.idx[k]];
        // Move the already solved off-block terms to the right-hand side:
        // M_ij = -theta*hh*J_ij, so r_i - M_ij x_j = r_i + theta*hh*J_ij x_j.
        if (!B.off_i.empty()) {
            for (size_t e = 0; e < B.off_i.size(); ++e) {
                int i = B.off_i[e], j = B.off_j[e];
                int k = (int)(std::lower_bound(B.idx.begin(), B.idx.end(), i) - B.idx.begin());
                rb[k] += theta * hh * Jf[i*NS+j] * dx[j];
            }
        }
        lu_solve(B.lu, B.piv, s, rb.data());
        for (int k = 0; k < s; ++k) dx[B.idx[k]] = rb[k];
    }
    t_solve += secs_since(a);
}

// Dense finite-difference probe at `at`: adds every new nonzero to the
// pattern and rebuilds the structure. Returns the number of new entries.
static int probe(double t1, const std::vector<double>& at, const char* why) {
    if (!BTF) return 0;
    auto a = clk::now(); long c0 = rhs_calls;
    std::vector<double> xb = at, f1(NS), fp(NS), xp;
    rhs(t1, xb, f1);
    int added = 0;
    for (int k = 0; k < NS; ++k) {
        xp = at; double d = 1e-7 * fmax(1.0, fabs(at[k])); xp[k] += d;
        rhs(t1, xp, fp);
        for (int i = 0; i < NS; ++i) if (fp[i] != f1[i] && !P[i*NS+k]) { P[i*NS+k] = 1; ++added; }
    }
    probe_rhs_calls += rhs_calls - c0; t_probe += secs_since(a);
    if (added) { printf("pattern grew by %d entries at t=%.9g (%s)\n", added, t1, why); build_structure(); }
    return added;
}

int main(int argc, char** argv) {
    double t_end = argc > 1 ? atof(argv[1]) : 0.01, h = argc > 2 ? atof(argv[2]) : 5e-6;
    const char* csv = argc > 3 ? argv[3] : nullptr;
    std::vector<double> y(NY), f0(NS), f1(NS), r(NS);
    int st = 0;
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    K(_load_starts_batch)(y.data(), p.data(), NY, NP, 1);
    K(_initialize_batch)(0.0, y.data(), p.data(), &st, NY, NP, 1);
    if (st) printf("init status %d\n", st);
    K(_delay_init_batch)(0.0, y.data(), p.data(), hist.data(), NY, NP, HSTRIDE, 1);
    printf("BTF=%d REUSE_JF=%d BLOCK_NEWTON=%d ATOL=%g RTOL=%g PROBE_EVERY=%d THETA=%g MAXDEPTH=%d\n", BTF, REUSE_JF, BLOCK_NEWTON, (double)NEWTON_ATOL, (double)NEWTON_RTOL, PROBE_EVERY, (double)THETA, MAXDEPTH);
    if (BTF) { for (int i = 0; i < NS; ++i) P[i*NS+i] = 1; probe(0.0, y, "initial"); }
    else { std::fill(P.begin(), P.end(), 1); build_structure(); }
    FILE* out = csv ? fopen(csv, "w") : nullptr;
    auto start = clk::now();
    double t = 0; int steps = (int)(t_end / h + 0.5), jac_age = 1 << 30, jacobians = 0, factorizations = 0, failed = 0, halvings = 0;
    double fact_h = -1, fact_theta = -1; bool jf_valid = false;
    rhs(t, y, f0);
    double theta = 0.5; long be_steps = 0; int fail_probes = 0, probes_periodic = 0;
    auto residual = [&](const std::vector<double>& x, const std::vector<double>& f1, double hh, std::vector<double>& out) {
        double n2 = 0; for (int j = 0; j < NS; ++j) { out[j] = -(x[j] - y[j] - hh * ((1.0 - theta) * f0[j] + theta * f1[j])); n2 += out[j] * out[j]; } return n2; };
    double best_err = 0; std::vector<double> best_x; long relaxed = 0;
    auto attempt = [&](double hh, std::vector<double>& x) -> bool {
        best_err = INFINITY;
        double t1 = t + hh;
        x = y; for (int j = 0; j < NS; ++j) x[j] = y[j] + hh * f0[j];
        for (int refresh = 0; refresh < 4; ++refresh) {
            bool new_jf = refresh > 0 || jac_age > 50 || !jf_valid || (!REUSE_JF && (fact_h != hh || fact_theta != theta));
            if (new_jf) {
                // Jacobian at the start point first, then at the current
                // iterate (a saturation breakpoint changes the slopes).
                const std::vector<double> at = refresh == 0 ? y : x;
                jacobian(t1, at, f1);
                jac_age = 0; jf_valid = true; ++jacobians;
            }
            if (new_jf || fact_h != hh || fact_theta != theta) {
                if (!factor(theta, hh)) { fact_h = -1; return false; }
                fact_h = hh; fact_theta = theta; ++factorizations;
            }
            std::vector<double> xt(NY), rr(NS);
            rhs(t1, x, f1); double n2 = residual(x, f1, hh, r);
#if BLOCK_NEWTON
            // Newton per independent group: own line search, own convergence.
            const int G = (int)ngroups.size();
            std::vector<double> gn2(G, 0.0), lam(G), gnew(G);
            std::vector<char> conv(G, 0), acc(G);
            for (int g = 0; g < G; ++g) for (int i : ngroups[g]) gn2[g] += r[i] * r[i];
            for (int it = 0; it < 10; ++it) {
                std::vector<double> dx = r; solve(theta, hh, dx);
                // A group whose full step is already below the tolerance takes
                // it without the decrease test: its residual is at round-off
                // level and need not decrease.
                std::vector<char> tiny(G, 0);
                for (int g = 0; g < G; ++g) {
                    lam[g] = 1.0; gnew[g] = 0.0; acc[g] = conv[g];
                    double norm = 0; for (int i : ngroups[g]) norm = fmax(norm, fabs(dx[i]) / (NEWTON_ATOL + NEWTON_RTOL * fabs(x[i] + dx[i])));
                    tiny[g] = G > 1 && norm < 1.0;
                }
                xt = x;
                for (int trial = 0; trial < 8; ++trial) {
                    for (int g = 0; g < G; ++g) if (!acc[g]) for (int i : ngroups[g]) xt[i] = x[i] + lam[g] * dx[i];
                    int st = rhs(t1, xt, f1);
                    residual(xt, f1, hh, rr);
                    bool all = true;
                    for (int g = 0; g < G; ++g) {
                        if (acc[g]) continue;
                        double n = 0; for (int i : ngroups[g]) n += rr[i] * rr[i];
                        gnew[g] = n;
                        if (!st && std::isfinite(n) && (n < gn2[g] || trial == 7 || tiny[g])) { acc[g] = 1; for (int i : ngroups[g]) r[i] = rr[i]; }
                        else { lam[g] *= 0.5; all = false; }
                    }
                    if (all) break;
                }
                bool bad = false;
                for (int g = 0; g < G; ++g) if (!conv[g] && !std::isfinite(gnew[g])) bad = true;
                if (bad) break;
                bool all_conv = true;
                for (int g = 0; g < G; ++g) {
                    if (conv[g]) continue;
                    double norm = 0; for (int i : ngroups[g]) norm = fmax(norm, fabs(lam[g] * dx[i]) / (NEWTON_ATOL + NEWTON_RTOL * fabs(xt[i])));
                    gn2[g] = gnew[g];
                    if (norm < 1.0 && lam[g] == 1.0) conv[g] = 1; else all_conv = false;
                }
                x = xt;
                { double e = 0; for (int j = 0; j < NS; ++j) e = fmax(e, fabs(r[j]) / (1.0 + fabs(x[j]))); if (e < best_err) { best_err = e; best_x = x; } }
                if (all_conv) return true;
            }
            (void)n2;
#else
            for (int it = 0; it < 10; ++it) {
                std::vector<double> dx = r; solve(theta, hh, dx);
                double lambda = 1.0, n2new = 0;
                for (int trial = 0; trial < 8; ++trial) {
                    xt = x; for (int j = 0; j < NS; ++j) xt[j] = x[j] + lambda * dx[j];
                    int st = rhs(t1, xt, f1);
                    n2new = residual(xt, f1, hh, rr);
                    if (!st && std::isfinite(n2new) && (n2new < n2 || trial == 7)) { r = rr; break; }
                    lambda *= 0.5;
                }
                if (!std::isfinite(n2new)) break;
                double norm = 0; for (int j = 0; j < NS; ++j) norm = fmax(norm, fabs(lambda * dx[j]) / (NEWTON_ATOL + NEWTON_RTOL * fabs(xt[j])));
                x = xt; n2 = n2new;
                { double e = 0; for (int j = 0; j < NS; ++j) e = fmax(e, fabs(r[j]) / (1.0 + fabs(x[j]))); if (e < best_err) { best_err = e; best_x = x; } }
                if (norm < 1.0 && lambda == 1.0) return true;
            }
#endif
        }
        return false;
    };
    std::function<bool(double, int)> advance = [&](double hh, int depth) -> bool {
        std::vector<double> x;
        theta = THETA;
        bool done = attempt(hh, x);
        if (!done && THETA < 1.0) {
            theta = 1.0; done = attempt(hh, x); be_steps += done;
        }
        if (!done && depth == 0 && fail_probes < MAX_FAIL_PROBES) {
            // A missing pattern entry gives a wrong Newton matrix: check the
            // pattern here before subdividing the step.
            ++fail_probes;
            if (probe(t + hh, y, "failed step") > 0) {
                jf_valid = false; fact_h = -1;
                theta = THETA; done = attempt(hh, x);
                if (!done && THETA < 1.0) { theta = 1.0; done = attempt(hh, x); be_steps += done; }
            }
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
        if (BTF && PROBE_EVERY > 0 && (n + 1) % PROBE_EVERY == 0) {
            ++probes_periodic;
            // The probe calls rhs at states it perturbs; it does not touch y.
            std::vector<double> at = y;
            if (probe(t + h, at, "periodic") > 0) { jf_valid = false; fact_h = -1; }
        }
    }
    double secs = secs_since(start);
    printf("rhs_failures=%ld\n", rhs_failures); printf("halvings=%d relaxed=%ld be_steps=%ld\n", halvings, relaxed, be_steps);
    printf("h=%g t=%g steps=%d unconverged=%d jacobians=%d factorizations=%d rhs_calls=%ld wall=%.2fs\n", h, t, steps, failed, jacobians, factorizations, rhs_calls, secs);
    printf("rhs_calls: jacobian=%ld probes=%ld other=%ld; probes: periodic=%d after_failure=%d; pattern builds=%d nnz=%ld\n",
           jac_rhs_calls, probe_rhs_calls, rhs_calls - jac_rhs_calls - probe_rhs_calls, probes_periodic, fail_probes, pattern_builds, pattern_nnz);
    printf("time: rhs(all)=%.2fs jacobian=%.2fs probes=%.2fs lu=%.2fs solve=%.2fs\n", t_rhs, t_jac, t_probe, t_lu, t_solve);
    printf("  "); for (int j = 0; j < 12; ++j) printf(" y[%d]=%.8g", j, y[j]); printf("\n");
    if (out) fclose(out);
}

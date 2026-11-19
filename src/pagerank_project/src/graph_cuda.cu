// CUDA PageRank and BFS. Build: nvcc -O3 -arch=sm_70 (needs sm_60+ for double atomicAdd).
//  PageRank: pull-based on in-CSR. Kernel chosen by mean in-degree: warp-per-vertex (skewed / high degree, e.g. RMAT)
//            or thread-per-vertex (low degree, e.g. road networks) so lanes are not wasted.
//  BFS: level-synchronous top-down with an explicit frontier queue, warp-per-frontier-vertex, atomicCAS to claim vertices.
#include "common.h"
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

__global__ void k_contrib(const double* r, const i64* off, double* c, double* dang, int n) {
  int u = blockIdx.x * blockDim.x + threadIdx.x; double dv = 0;
  if (u < n) { i64 od = off[u + 1] - off[u]; if (od) c[u] = r[u] / od; else { c[u] = 0; dv = r[u]; } }
  for (int o = 16; o > 0; o >>= 1) dv += __shfl_down_sync(0xffffffff, dv, o);
  if ((threadIdx.x & 31) == 0 && dv != 0) atomicAdd(dang, dv);
}
__global__ void k_pull_warp(const i64* roff, const i32* radj, const double* c, const double* r, double* rn,
                            const double* dang, double* diff, int n, double d) {
  __shared__ double sd; if (threadIdx.x == 0) sd = 0; __syncthreads();
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; double ld = 0;
  if (gw < n) {
    double s = 0; for (i64 e = roff[gw] + lane; e < roff[gw + 1]; e += 32) s += c[radj[e]];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) { double nv = (1 - d) / n + d * (s + *dang / n); rn[gw] = nv; ld = fabs(nv - r[gw]); }
  }
  if (ld != 0) atomicAdd(&sd, ld); __syncthreads(); if (threadIdx.x == 0) atomicAdd(diff, sd);
}
__global__ void k_pull_thread(const i64* roff, const i32* radj, const double* c, const double* r, double* rn,
                              const double* dang, double* diff, int n, double d) {
  __shared__ double sd; if (threadIdx.x == 0) sd = 0; __syncthreads();
  int v = blockIdx.x * blockDim.x + threadIdx.x; double ld = 0;
  if (v < n) { double s = 0; for (i64 e = roff[v]; e < roff[v + 1]; e++) s += c[radj[e]];
    double nv = (1 - d) / n + d * (s + *dang / n); rn[v] = nv; ld = fabs(nv - r[v]); }
  if (ld != 0) atomicAdd(&sd, ld); __syncthreads(); if (threadIdx.x == 0) atomicAdd(diff, sd);
}
__global__ void k_bfs(const i64* off, const i32* adj, i32* dist, const i32* fin, int nf, i32* fout, int* nout, int level) {
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (gw >= nf) return;
  i32 u = fin[gw];
  for (i64 e = off[u] + lane; e < off[u + 1]; e += 32) {
    i32 v = adj[e];
    if (dist[v] == -1 && atomicCAS(&dist[v], -1, level + 1) == -1) fout[atomicAdd(nout, 1)] = v;
  }
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); Graph g;
  if (!load_graph(a.graph.c_str(), g)) { fprintf(stderr, "cannot load\n"); return 1; }
  const int n = g.n; i64 *d_off, *d_roff; i32 *d_adj, *d_radj;
  CK(cudaMalloc(&d_off, (n + 1) * 8)); CK(cudaMalloc(&d_roff, (n + 1) * 8));
  CK(cudaMalloc(&d_adj, g.m * 4)); CK(cudaMalloc(&d_radj, g.m * 4));
  CK(cudaMemcpy(d_off, g.off.data(), (n + 1) * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_roff, g.roff.data(), (n + 1) * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_adj, g.adj.data(), g.m * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_radj, g.radj.data(), g.m * 4, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); printf("[cuda] %s on %s n=%d m=%lld\n", a.name.c_str(), pr.name, n, (long long)g.m);
  const int B = 256;
  if (a.algo != "bfs") {
    double *d_r, *d_rn, *d_c, *d_s; CK(cudaMalloc(&d_r, n * 8)); CK(cudaMalloc(&d_rn, n * 8)); CK(cudaMalloc(&d_c, n * 8)); CK(cudaMalloc(&d_s, 16));
    bool warp = (double)g.m / n >= 8.0; std::vector<double> init(n, 1.0 / n), hist, r(n);
    for (int k = 0; k < a.reps; k++) {
      CK(cudaMemcpy(d_r, init.data(), n * 8, cudaMemcpyHostToDevice)); hist.clear(); int it = 0; double diff;
      CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
      do {
        CK(cudaMemset(d_s, 0, 16));
        k_contrib<<<(n + B - 1) / B, B>>>(d_r, d_off, d_c, d_s, n);
        if (warp) k_pull_warp<<<(int)(((i64)n * 32 + B - 1) / B), B>>>(d_roff, d_radj, d_c, d_r, d_rn, d_s, d_s + 1, n, DAMP);
        else      k_pull_thread<<<(n + B - 1) / B, B>>>(d_roff, d_radj, d_c, d_r, d_rn, d_s, d_s + 1, n, DAMP);
        CK(cudaMemcpy(&diff, d_s + 1, 8, cudaMemcpyDeviceToHost));   // implicit sync; also the convergence test
        std::swap(d_r, d_rn); it++; hist.push_back(diff);
      } while (diff >= TOL && it < MAXIT);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      emit_row(a, "cuda", "pagerank", g, 1, k, it, ms / 1e3, (double)g.m * it, warp ? "kernel=warp_per_vertex" : "kernel=thread_per_vertex");
      printf("  PR run %d: %d iters, %.4f s, %.3f ms/iter\n", k, it, ms / 1e3, ms / it);
      if (k == 0) { emit_conv(a, "cuda", g, 1, hist); CK(cudaMemcpy(r.data(), d_r, n * 8, cudaMemcpyDeviceToHost)); dump_ranks(a.dump, r); }
    }
  }
  if (a.algo != "pagerank") {
    auto roots = select_roots(g, a.roots); std::vector<i32> dist(n);
    i32 *d_dist, *fa, *fb; int* d_cnt; CK(cudaMalloc(&d_dist, n * 4)); CK(cudaMalloc(&fa, n * 4)); CK(cudaMalloc(&fb, n * 4)); CK(cudaMalloc(&d_cnt, 4));
    for (size_t k = 0; k < roots.size(); k++) {
      CK(cudaMemset(d_dist, 0xFF, n * 4)); i32 z = 0; i32 rt = roots[k];
      CK(cudaMemcpy(d_dist + rt, &z, 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(fa, &rt, 4, cudaMemcpyHostToDevice));
      int nf = 1, level = 0; CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
      while (nf > 0) {
        CK(cudaMemset(d_cnt, 0, 4));
        k_bfs<<<(int)(((i64)nf * 32 + B - 1) / B), B>>>(d_off, d_adj, d_dist, fa, nf, fb, d_cnt, level);
        CK(cudaMemcpy(&nf, d_cnt, 4, cudaMemcpyDeviceToHost)); std::swap(fa, fb); level++;
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      CK(cudaMemcpy(dist.data(), d_dist, n * 4, cudaMemcpyDeviceToHost)); i64 E = edges_traversed(g, dist);
      emit_row(a, "cuda", "bfs", g, 1, (int)k, level, ms / 1e3, (double)E);
      printf("  BFS root %d: %d levels, %.4f s, %.1f MTEPS\n", rt, level, ms / 1e3, E / (ms / 1e3) / 1e6);
    }
  }
}

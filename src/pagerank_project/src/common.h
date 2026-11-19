// Shared graph loader, CSV writer, root selection. Used by OpenMP, MPI and CUDA programs.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
typedef int64_t i64; typedef int32_t i32;

static const double DAMP = 0.85, TOL = 1e-6;
static const int MAXIT = 200;

struct Graph {                       // out-CSR (adj) and in-CSR (radj)
  i64 n = 0, m = 0;
  std::vector<i64> off, roff;
  std::vector<i32> adj, radj;
  i64 outdeg(i32 u) const { return off[u + 1] - off[u]; }
};

inline bool load_graph(const char* path, Graph& g) {
  FILE* f = fopen(path, "rb"); if (!f) return false;
  char magic[8]; i64 h[2];
  if (fread(magic, 1, 8, f) != 8 || fread(h, 8, 2, f) != 2) return false;
  g.n = h[0]; g.m = h[1];
  g.off.resize(g.n + 1); g.adj.resize(g.m);
  if (fread(g.off.data(), 8, g.n + 1, f) != (size_t)g.n + 1) return false;
  if (fread(g.adj.data(), 4, g.m, f) != (size_t)g.m) return false;
  fclose(f);
  g.roff.assign(g.n + 1, 0);                       // transpose -> in-edges for pull-style PageRank
  for (i64 e = 0; e < g.m; e++) g.roff[g.adj[e] + 1]++;
  for (i64 i = 0; i < g.n; i++) g.roff[i + 1] += g.roff[i];
  g.radj.resize(g.m);
  std::vector<i64> pos(g.roff.begin(), g.roff.end() - 1);
  for (i64 u = 0; u < g.n; u++)
    for (i64 e = g.off[u]; e < g.off[u + 1]; e++) g.radj[pos[g.adj[e]]++] = (i32)u;
  return true;
}

// Serial BFS used only to choose roots (identical on every platform) and to validate.
inline i64 serial_bfs_reach(const Graph& g, i32 root, std::vector<i32>& dist) {
  std::fill(dist.begin(), dist.end(), -1); dist[root] = 0;
  std::vector<i32> q; q.push_back(root); i64 cnt = 1;
  for (size_t h = 0; h < q.size(); h++) {
    i32 u = q[h];
    for (i64 e = g.off[u]; e < g.off[u + 1]; e++) { i32 v = g.adj[e]; if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push_back(v); cnt++; } }
  }
  return cnt;
}
// k deterministic roots whose BFS reaches >=10% of the graph (Graph500-style "non-trivial" roots)
inline std::vector<i32> select_roots(const Graph& g, int k, unsigned seed = 12345) {
  std::mt19937_64 rng(seed); std::vector<i32> roots, dist(g.n);
  for (int tries = 0; (int)roots.size() < k && tries < 5000; tries++) {
    i32 u = rng() % g.n; if (g.outdeg(u) == 0) continue;
    if (serial_bfs_reach(g, u, dist) >= g.n / 10) roots.push_back(u);
  }
  return roots;
}
inline i64 edges_traversed(const Graph& g, const std::vector<i32>& dist) {  // sum of out-degrees of reached vertices
  i64 s = 0; for (i64 u = 0; u < g.n; u++) if (dist[u] >= 0) s += g.outdeg((i32)u); return s;
}

struct Args {
  std::string graph, name = "g", type = "unknown", out = "results.csv", conv = "", dump = "", algo = "both";
  int threads = 1, reps = 3, roots = 8;
};
inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "-g") a.graph = v; else if (k == "-n") a.name = v; else if (k == "-y") a.type = v;
    else if (k == "-o") a.out = v; else if (k == "-c") a.conv = v; else if (k == "-d") a.dump = v;
    else if (k == "-a") a.algo = v; else if (k == "-T") a.threads = atoi(v.c_str());
    else if (k == "-r") a.reps = atoi(v.c_str()); else if (k == "-R") a.roots = atoi(v.c_str());
  }
  return a;
}

// ---- common CSV format (one row per timed run) ----
static const char* CSV_HEADER = "platform,algorithm,graph,graph_type,n,m,workers,run,iterations,time_s,time_per_iter_ms,edges_processed,teps,notes";
inline void emit_row(const Args& a, const char* platform, const char* algo, const Graph& g, int workers, int run,
                     int iters, double secs, double edges, const std::string& notes = "") {
  FILE* t = fopen(a.out.c_str(), "r"); bool fresh = (t == nullptr); if (t) fclose(t);
  FILE* f = fopen(a.out.c_str(), "a"); if (!f) return;
  if (fresh) fprintf(f, "%s\n", CSV_HEADER);
  fprintf(f, "%s,%s,%s,%s,%lld,%lld,%d,%d,%d,%.6f,%.6f,%.0f,%.6e,%s\n", platform, algo, a.name.c_str(), a.type.c_str(),
          (long long)g.n, (long long)g.m, workers, run, iters, secs, 1e3 * secs / iters, edges, edges / secs, notes.c_str());
  fclose(f);
}
inline void emit_conv(const Args& a, const char* platform, const Graph& g, int workers, const std::vector<double>& hist) {
  if (a.conv.empty()) return;
  FILE* t = fopen(a.conv.c_str(), "r"); bool fresh = (t == nullptr); if (t) fclose(t);
  FILE* f = fopen(a.conv.c_str(), "a"); if (!f) return;
  if (fresh) fprintf(f, "platform,graph,workers,iter,residual_l1\n");
  for (size_t i = 0; i < hist.size(); i++) fprintf(f, "%s,%s,%d,%zu,%.6e\n", platform, a.name.c_str(), workers, i + 1, hist[i]);
  fclose(f);
}
inline void dump_ranks(const std::string& p, const std::vector<double>& r) {
  if (p.empty()) return; FILE* f = fopen(p.c_str(), "wb"); fwrite(r.data(), 8, r.size(), f); fclose(f);
}

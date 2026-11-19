// OpenMP PageRank (pull-based, in-CSR) and level-synchronous top-down BFS.
#include "common.h"
#include <omp.h>

static int pagerank(const Graph& g, int T, std::vector<double>& r, std::vector<double>& hist, double& secs) {
  const i64 n = g.n; r.assign(n, 1.0 / n);
  std::vector<double> rn(n), c(n); hist.clear();
  double t0 = omp_get_wtime(), diff; int it = 0;
  do {
    double dang = 0;
    #pragma omp parallel for num_threads(T) reduction(+:dang) schedule(static)
    for (i64 u = 0; u < n; u++) {
      i64 od = g.off[u + 1] - g.off[u];
      if (od) c[u] = r[u] / od; else { c[u] = 0; dang += r[u]; }   // dangling mass redistributed uniformly
    }
    const double base = (1 - DAMP) / n + DAMP * dang / n; diff = 0;
    #pragma omp parallel for num_threads(T) reduction(+:diff) schedule(dynamic, 1024)
    for (i64 v = 0; v < n; v++) {
      double s = 0;
      for (i64 e = g.roff[v]; e < g.roff[v + 1]; e++) s += c[g.radj[e]];
      double nv = base + DAMP * s; diff += fabs(nv - r[v]); rn[v] = nv;
    }
    r.swap(rn); it++; hist.push_back(diff);
  } while (diff >= TOL && it < MAXIT);
  secs = omp_get_wtime() - t0; return it;
}

static int bfs(const Graph& g, i32 root, int T, std::vector<i32>& dist, double& secs) {
  std::fill(dist.begin(), dist.end(), -1); dist[root] = 0;
  std::vector<i32> front{root}; std::vector<std::vector<i32>> loc(T); int level = 0;
  double t0 = omp_get_wtime();
  while (!front.empty()) {
    #pragma omp parallel num_threads(T)
    {
      auto& nb = loc[omp_get_thread_num()]; nb.clear();
      #pragma omp for schedule(dynamic, 64)
      for (size_t i = 0; i < front.size(); i++) {
        i32 u = front[i];
        for (i64 e = g.off[u]; e < g.off[u + 1]; e++) {
          i32 v = g.adj[e]; i32 exp = -1;
          if (dist[v] == -1 && __atomic_compare_exchange_n(&dist[v], &exp, level + 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            nb.push_back(v);
        }
      }
    }
    front.clear(); for (auto& nb : loc) front.insert(front.end(), nb.begin(), nb.end());
    level++;
  }
  secs = omp_get_wtime() - t0; return level;
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); Graph g;
  if (!load_graph(a.graph.c_str(), g)) { fprintf(stderr, "cannot load %s\n", a.graph.c_str()); return 1; }
  int T = a.threads; printf("[omp] %s n=%lld m=%lld threads=%d\n", a.name.c_str(), (long long)g.n, (long long)g.m, T);
  if (a.algo != "bfs") {
    std::vector<double> r, hist;
    for (int k = 0; k < a.reps; k++) {
      double s; int it = pagerank(g, T, r, hist, s);
      emit_row(a, "openmp", "pagerank", g, T, k, it, s, (double)g.m * it);
      printf("  PR run %d: %d iters, %.4f s, %.3f ms/iter\n", k, it, s, 1e3 * s / it);
      if (k == 0) { emit_conv(a, "openmp", g, T, hist); dump_ranks(a.dump, r); }
    }
  }
  if (a.algo != "pagerank") {
    auto roots = select_roots(g, a.roots); std::vector<i32> dist(g.n);
    for (size_t k = 0; k < roots.size(); k++) {
      double s; int lv = bfs(g, roots[k], T, dist, s); i64 E = edges_traversed(g, dist);
      emit_row(a, "openmp", "bfs", g, T, (int)k, lv, s, (double)E);
      printf("  BFS root %d: %d levels, %.4f s, %.1f MTEPS\n", roots[k], lv, s, E / s / 1e6);
    }
  }
}

// MPI PageRank and BFS. 1-D vertex partition balanced on (in-degree+out-degree+1). Every rank loads the CSR
// (simple, memory-replicated) but only processes its own rows; the rank vector / frontier is exchanged with Allgatherv.
#include "common.h"
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv); int P, me; MPI_Comm_size(MPI_COMM_WORLD, &P); MPI_Comm_rank(MPI_COMM_WORLD, &me);
  Args a = parse_args(argc, argv); Graph g;
  if (!load_graph(a.graph.c_str(), g)) { if (!me) fprintf(stderr, "cannot load\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
  const i64 n = g.n;
  std::vector<i64> bnd(P + 1, 0); {                       // partition bounds
    double tot = 0; for (i64 v = 0; v < n; v++) tot += (g.roff[v + 1] - g.roff[v]) + g.outdeg(v) + 1;
    double acc = 0; int p = 1;
    for (i64 v = 0; v < n && p < P; v++) { acc += (g.roff[v + 1] - g.roff[v]) + g.outdeg(v) + 1; if (acc >= tot * p / P) bnd[p++] = v + 1; }
    while (p < P) bnd[p++] = n; bnd[P] = n;
  }
  const i64 lo = bnd[me], hi = bnd[me + 1];
  std::vector<int> cnt(P), dsp(P); for (int p = 0; p < P; p++) { cnt[p] = bnd[p + 1] - bnd[p]; dsp[p] = bnd[p]; }
  if (!me) printf("[mpi] %s n=%lld m=%lld ranks=%d\n", a.name.c_str(), (long long)n, (long long)g.m, P);

  if (a.algo != "bfs") {
    std::vector<double> r(n), c(n), mine(hi - lo), hist;
    for (int k = 0; k < a.reps; k++) {
      std::fill(r.begin(), r.end(), 1.0 / n); hist.clear(); double comm = 0, diff; int it = 0;
      MPI_Barrier(MPI_COMM_WORLD); double t0 = MPI_Wtime();
      do {
        double dang = 0;
        for (i64 u = 0; u < n; u++) { i64 od = g.off[u + 1] - g.off[u]; if (od) c[u] = r[u] / od; else { c[u] = 0; dang += r[u]; } }
        const double base = (1 - DAMP) / n + DAMP * dang / n; double ld = 0;
        for (i64 v = lo; v < hi; v++) {
          double s = 0; for (i64 e = g.roff[v]; e < g.roff[v + 1]; e++) s += c[g.radj[e]];
          double nv = base + DAMP * s; ld += fabs(nv - r[v]); mine[v - lo] = nv;
        }
        double tc = MPI_Wtime();
        MPI_Allgatherv(mine.data(), hi - lo, MPI_DOUBLE, r.data(), cnt.data(), dsp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allreduce(&ld, &diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm += MPI_Wtime() - tc; it++; hist.push_back(diff);
      } while (diff >= TOL && it < MAXIT);
      double s = MPI_Wtime() - t0; MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (!me) {
        char note[64]; snprintf(note, 64, "comm_frac=%.3f", comm / s);
        emit_row(a, "mpi", "pagerank", g, P, k, it, s, (double)g.m * it, note);
        printf("  PR run %d: %d iters, %.4f s, %.3f ms/iter, %s\n", k, it, s, 1e3 * s / it, note);
        if (k == 0) { emit_conv(a, "mpi", g, P, hist); dump_ranks(a.dump, r); }
      }
    }
  }
  if (a.algo != "pagerank") {
    std::vector<i32> roots; int nr = 0;
    if (!me) { roots = select_roots(g, a.roots); nr = roots.size(); }
    MPI_Bcast(&nr, 1, MPI_INT, 0, MPI_COMM_WORLD); roots.resize(nr); MPI_Bcast(roots.data(), nr, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<i32> dist(n), mark(n, -1); int stamp = 0;
    for (int k = 0; k < nr; k++) {
      std::fill(dist.begin(), dist.end(), -1); dist[roots[k]] = 0;
      std::vector<i32> lf, nlf, cand, all; if (roots[k] >= lo && roots[k] < hi) lf.push_back(roots[k]);
      int level = 0; double comm = 0;
      MPI_Barrier(MPI_COMM_WORLD); double t0 = MPI_Wtime();
      while (true) {
        stamp++; cand.clear();
        for (i32 u : lf) for (i64 e = g.off[u]; e < g.off[u + 1]; e++) { i32 v = g.adj[e]; if (dist[v] < 0 && mark[v] != stamp) { mark[v] = stamp; cand.push_back(v); } }
        double tc = MPI_Wtime(); int mc = cand.size(); std::vector<int> cc(P), dd(P);
        MPI_Allgather(&mc, 1, MPI_INT, cc.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int tot = 0; for (int p = 0; p < P; p++) { dd[p] = tot; tot += cc[p]; } all.resize(tot);
        MPI_Allgatherv(cand.data(), mc, MPI_INT, all.data(), cc.data(), dd.data(), MPI_INT, MPI_COMM_WORLD);
        comm += MPI_Wtime() - tc; nlf.clear(); int found = 0;
        for (i32 v : all) if (dist[v] < 0) { dist[v] = level + 1; found++; if (v >= lo && v < hi) nlf.push_back(v); }
        lf.swap(nlf); level++; if (!found) break;
      }
      double s = MPI_Wtime() - t0; MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (!me) {
        i64 E = edges_traversed(g, dist); char note[64]; snprintf(note, 64, "comm_frac=%.3f", comm / s);
        emit_row(a, "mpi", "bfs", g, P, k, level, s, (double)E, note);
        printf("  BFS root %d: %d levels, %.4f s, %.1f MTEPS, %s\n", roots[k], level, s, E / s / 1e6, note);
      }
    }
  }
  MPI_Finalize();
}

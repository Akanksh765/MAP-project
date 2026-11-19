// Graph generators.  gen rmat SCALE EDGEFACTOR SEED OUT   (Graph500 RMAT: a=.57 b=.19 c=.19 d=.05, directed, labels permuted)
//                    gen grid SIDE SEED OUT               (road-network proxy: 2D lattice, 10% edges removed, bidirectional)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
typedef int64_t i64; typedef uint64_t u64;
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage\n"); return 1; }
  std::string kind = argv[1]; std::vector<u64> keys; i64 n;
  const char* out;
  if (kind == "rmat") {
    int scale = atoi(argv[2]), ef = atoi(argv[3]); std::mt19937_64 rng(atoll(argv[4])); out = argv[5];
    n = 1LL << scale; i64 m = n * ef; std::vector<uint32_t> perm(n);
    for (i64 i = 0; i < n; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng);
    std::uniform_real_distribution<double> U(0, 1); keys.reserve(m);
    for (i64 e = 0; e < m; e++) {
      u64 u = 0, v = 0;
      for (int b = 0; b < scale; b++) { double r = U(rng); u <<= 1; v <<= 1;
        if (r < .57) {} else if (r < .76) v |= 1; else if (r < .95) u |= 1; else { u |= 1; v |= 1; } }
      u = perm[u]; v = perm[v]; if (u != v) keys.push_back(u << 32 | v);
    }
  } else {
    int S = atoi(argv[2]); std::mt19937_64 rng(atoll(argv[3])); out = argv[4]; n = (i64)S * S;
    std::uniform_real_distribution<double> U(0, 1);
    for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
      u64 a = (u64)y * S + x;
      if (x + 1 < S && U(rng) < .9) { u64 b = a + 1; keys.push_back(a << 32 | b); keys.push_back(b << 32 | a); }
      if (y + 1 < S && U(rng) < .9) { u64 b = a + S; keys.push_back(a << 32 | b); keys.push_back(b << 32 | a); }
    }
  }
  std::sort(keys.begin(), keys.end()); keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  i64 m = keys.size(); std::vector<i64> off(n + 1, 0); std::vector<int32_t> adj(m);
  for (i64 e = 0; e < m; e++) { off[(keys[e] >> 32) + 1]++; adj[e] = (int32_t)(keys[e] & 0xffffffffu); }
  for (i64 i = 0; i < n; i++) off[i + 1] += off[i];
  FILE* f = fopen(out, "wb"); fwrite("PRGRAPH1", 1, 8, f); i64 h[2] = {n, m}; fwrite(h, 8, 2, f);
  fwrite(off.data(), 8, n + 1, f); fwrite(adj.data(), 4, m, f); fclose(f);
  printf("%s: n=%lld m=%lld\n", out, (long long)n, (long long)m);
}

import numpy as np, sys, subprocess
from scipy.sparse import csr_matrix
def load(p):
    with open(p,'rb') as f:
        f.read(8); n,m=np.frombuffer(f.read(16),dtype=np.int64)
        off=np.frombuffer(f.read(8*(n+1)),dtype=np.int64); adj=np.frombuffer(f.read(4*m),dtype=np.int32)
    return int(n),int(m),off,adj
def ref(p,d=.85,tol=1e-12):
    n,m,off,adj=load(p); src=np.repeat(np.arange(n),np.diff(off))
    A=csr_matrix((np.ones(m),(adj,src)),shape=(n,n)); od=np.diff(off).astype(float); r=np.full(n,1/n)
    for _ in range(1000):
        c=np.divide(r,od,out=np.zeros(n),where=od>0); dang=r[od==0].sum()
        rn=(1-d)/n+d*(A@c+dang/n); 
        if abs(rn-r).sum()<tol: r=rn;break
        r=rn
    return r
bad=0
for g in sys.argv[1:]:
    r0=ref(f'data/{g}.bin')
    for plat in ['omp','mpi','cuda']:
        try: x=np.fromfile(f'results/dump_{plat}_{g}.bin')
        except FileNotFoundError: continue
        l1=abs(x-r0).sum(); ok=l1<5e-5 and abs(x.sum()-1)<1e-9
        print(f'{g:10s} {plat:5s} L1 vs scipy-reference = {l1:.2e}  sum={x.sum():.12f}  {"PASS" if ok else "FAIL"}'); bad+=not ok
sys.exit(bad)

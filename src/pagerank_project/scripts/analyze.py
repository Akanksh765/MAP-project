import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
os.chdir(os.path.join(os.path.dirname(__file__),'..'))
d=pd.read_csv('results/all_results.csv'); cv=pd.read_csv('results/convergence.csv')
d['comm']=d.notes.str.extract(r'comm_frac=([\d.]+)').astype(float)
GR=['rmat18','road512','rmat20','road1024']; LBL={'rmat18':'RMAT-18 (power-law)','road512':'Road 512² (lattice)','rmat20':'RMAT-20 (power-law)','road1024':'Road 1024² (lattice)'}
COL={'openmp':'#1f77b4','mpi':'#d62728','cuda':'#2ca02c'}
plats=sorted(d.platform.unique()); has_cuda='cuda' in plats
# ---------- aggregate ----------
pr=d[d.algorithm=='pagerank'].groupby(['graph','platform','workers']).agg(iters=('iterations','first'),ms_iter=('time_per_iter_ms','median'),t=('time_s','median'),m=('m','first'),n=('n','first')).reset_index()
pr['ns_edge']=pr.ms_iter*1e6/pr.m; pr['MEPS']=pr.m/(pr.ms_iter*1e-3)/1e6
b=d[d.algorithm=='bfs'].groupby(['graph','platform','workers']).agg(levels=('iterations','median'),t=('time_s','sum'),E=('edges_processed','sum'),comm=('comm','median'),nroots=('run','count')).reset_index()
b['MTEPS']=b.E/b.t/1e6; b['ms_level']=1e3*b.t/(b.levels*b.nroots)
# harmonic-mean TEPS (Graph500 convention) equals E_total/t_total when each root has equal E; we use total ratio.
base_pr=pr[(pr.platform=='openmp')&(pr.workers==1)].set_index('graph').ms_iter
base_bfs=b[(b.platform=='openmp')&(b.workers==1)].set_index('graph').t
pr['speedup']=pr.apply(lambda r: base_pr[r.graph]/r.ms_iter,axis=1); b['speedup']=b.apply(lambda r: base_bfs[r.graph]/r.t,axis=1)
pr.to_csv('results/summary_pagerank.csv',index=False); b.to_csv('results/summary_bfs.csv',index=False)
# ---------- Fig 1 convergence ----------
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
cc={'rmat18':'#9467bd','rmat20':'#6a3d9a','road512':'#ff7f0e','road1024':'#b35806'}
for g in GR:
    o=cv[(cv.graph==g)&(cv.platform=='openmp')&(cv.workers==1)]
    ax[0].semilogy(o.iter,o.residual_l1,'-',color=cc[g],label=LBL[g],lw=2)
    mp=cv[(cv.graph==g)&(cv.platform=='mpi')&(cv.workers==cv[cv.platform=='mpi'].workers.max())]
    ax[0].semilogy(mp.iter,mp.residual_l1,'o',color=cc[g],ms=4,mfc='none')
ax[0].axhline(1e-6,color='k',ls='--',lw=1); ax[0].text(1,1.3e-6,'tolerance 1e-6',fontsize=8)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('L1 residual  ‖r_k − r_{k−1}‖₁'); ax[0].set_title('PageRank convergence (d = 0.85)\nlines: OpenMP, circles: MPI (4 ranks) – identical'); ax[0].legend(fontsize=8)
# empirical contraction factor over the last 5 iterations
for g in GR:
    o=cv[(cv.graph==g)&(cv.platform=='openmp')&(cv.workers==1)].residual_l1.values
    ax[1].bar(LBL[g].split(' (')[0],(o[-1]/o[-6])**(1/5),color=cc[g])
ax[1].axhline(0.85,color='k',ls='--',lw=1); ax[1].text(-0.4,0.855,'theoretical worst case = d = 0.85',fontsize=8)
ax[1].set_ylim(0,1); ax[1].set_ylabel('residual ratio per iteration (last 5 iters)'); ax[1].set_title('Asymptotic contraction rate'); plt.setp(ax[1].get_xticklabels(),rotation=15)
plt.tight_layout(); plt.savefig('figures/fig1_convergence.png',dpi=150); plt.close()
contr={g:(lambda o:(o[-1]/o[-6])**(1/5))(cv[(cv.graph==g)&(cv.platform=='openmp')&(cv.workers==1)].residual_l1.values) for g in GR}
# ---------- Fig 2 time per iteration ----------
def sty(c):
    p,w_=c.split('-'); plat={'OMP':'openmp','MPI':'mpi','CUDA':'cuda'}[p]
    return COL[plat],{'1':0.45,'2':0.7,'4':1.0}.get(w_,1.0)

def cfg(r): return f"{'OMP' if r.platform=='openmp' else r.platform.upper()}-{r.workers}"
pr['cfg']=pr.apply(cfg,axis=1); order=[c for c in ['OMP-1','OMP-2','OMP-4','MPI-1','MPI-2','MPI-4','CUDA-1'] if c in set(pr.cfg)]
fig,ax=plt.subplots(1,2,figsize=(13,4.4)); w=0.8/len(order)
for j,c in enumerate(order):
    for k,(col,yl) in enumerate([('ms_iter','time per iteration (ms)'),('ns_edge','ns per edge per iteration')]):
        vals=[pr[(pr.graph==g)&(pr.cfg==c)][col].values[0] for g in GR]
        ax[k].bar(np.arange(4)+j*w-0.4+w/2,vals,w,label=c,color=sty(c)[0],alpha=sty(c)[1])
for k,yl in enumerate(['time per iteration (ms, log)','ns per edge per iteration (log)']):
    ax[k].set_yscale('log'); ax[k].set_xticks(range(4)); ax[k].set_xticklabels([LBL[g].replace(' (','\n(') for g in GR],fontsize=8); ax[k].set_ylabel(yl)
ax[0].legend(ncol=3,fontsize=7); ax[0].set_title('PageRank time per iteration'+('' if has_cuda else '\n(CUDA not measured: no GPU on this host)')); ax[1].set_title('Normalised by edges')
plt.tight_layout(); plt.savefig('figures/fig2_time_per_iteration.png',dpi=150); plt.close()
# ---------- Fig 3 TEPS and speedup ----------
b['cfg']=b.apply(cfg,axis=1)
fig,ax=plt.subplots(2,2,figsize=(13,8))
for k,(df,col,yl,tt) in enumerate([(b,'MTEPS','BFS  MTEPS','BFS throughput (TEPS)'),(pr,'MEPS','PageRank  M edge-updates/s','PageRank throughput (edges×iters / time)')]):
    for j,c in enumerate(order):
        vals=[df[(df.graph==g)&(df.cfg==c)][col].values[0] if len(df[(df.graph==g)&(df.cfg==c)]) else np.nan for g in GR]
        ax[0][k].bar(np.arange(4)+j*w-0.4+w/2,vals,w,label=c,color=sty(c)[0],alpha=sty(c)[1])
    ax[0][k].set_xticks(range(4)); ax[0][k].set_xticklabels([LBL[g].replace(' (','\n(') for g in GR],fontsize=8); ax[0][k].set_ylabel(yl); ax[0][k].set_title(tt)
    for j,c in enumerate(order):
        vals=[df[(df.graph==g)&(df.cfg==c)]['speedup'].values[0] if len(df[(df.graph==g)&(df.cfg==c)]) else np.nan for g in GR]
        ax[1][k].bar(np.arange(4)+j*w-0.4+w/2,vals,w,color=sty(c)[0],alpha=sty(c)[1])
    ax[1][k].axhline(1,color='k',lw=1,ls='--'); ax[1][k].set_xticks(range(4)); ax[1][k].set_xticklabels([LBL[g].replace(' (','\n(') for g in GR],fontsize=8)
    ax[1][k].set_ylabel('speedup vs OpenMP, 1 thread'); ax[1][k].set_title('Speedup ('+('BFS' if k==0 else 'PageRank')+')  – host has 1 core: >1 is not expected')
ax[0][0].legend(ncol=3,fontsize=7); plt.tight_layout(); plt.savefig('figures/fig3_teps_speedup.png',dpi=150); plt.close()
# ---------- Fig 4 structure effect: levels, ms/level, comm fraction ----------
fig,ax=plt.subplots(1,3,figsize=(14,4))
x=np.arange(4); t1=b[(b.platform=='openmp')&(b.workers==1)].set_index('graph')
ax[0].bar(x,[t1.levels[g] for g in GR],color=[cc[g] for g in GR]); ax[0].set_yscale('log'); ax[0].set_title('BFS levels (≈ diameter)\n= number of global synchronisations'); ax[0].set_xticks(x); ax[0].set_xticklabels([g for g in GR],fontsize=8)
ax[1].bar(x,[t1.ms_level[g]*1e3 for g in GR],color=[cc[g] for g in GR]); ax[1].set_yscale('log'); ax[1].set_title('BFS time per level, OpenMP-1 (µs)'); ax[1].set_xticks(x); ax[1].set_xticklabels(GR,fontsize=8)
mm=b[b.platform=='mpi']
for g in GR:
    s=mm[mm.graph==g].sort_values('workers'); ax[2].plot(s.workers,s.comm,'o-',color=cc[g],label=g+' BFS')
    s2=d[(d.algorithm=='pagerank')&(d.platform=='mpi')&(d.graph==g)].groupby('workers').comm.median(); ax[2].plot(s2.index,s2.values,'s--',color=cc[g],alpha=.6,label=g+' PR')
ax[2].set_xlabel('MPI ranks'); ax[2].set_ylabel('fraction of time in MPI collectives'); ax[2].set_title('MPI communication share'); ax[2].legend(fontsize=6,ncol=2); ax[2].set_xticks([1,2,4])
plt.tight_layout(); plt.savefig('figures/fig4_structure_effects.png',dpi=150); plt.close()
# ---------- Tables ----------
def deg(g):
    with open(f'data/{g}.bin','rb') as f:
        f.read(8); n,m=np.frombuffer(f.read(16),dtype=np.int64); off=np.frombuffer(f.read(8*(n+1)),dtype=np.int64); adj=np.frombuffer(f.read(4*m),dtype=np.int32)
    od=np.diff(off); idg=np.bincount(adj,minlength=n); return n,m,od.max(),idg.max(),(od==0).mean()*100
rows=[]
for g in GR:
    n,m,mo,mi,dang=deg(g); rows.append(dict(graph=g,type='power-law' if 'rmat' in g else 'road-like',n=n,m=m,avg_deg=round(m/n,2),max_out=mo,max_in=mi,dangling_pct=round(dang,1),pr_iters=int(pr[pr.graph==g].iters.iloc[0]),bfs_levels=int(t1.levels[g]),contraction=round(contr[g],3)))
gs=pd.DataFrame(rows); gs.to_csv('results/graph_stats.csv',index=False)
with open('results/summary_tables.md','w') as f:
    f.write('## Graph statistics\n'+gs.to_markdown(index=False)+'\n\n## PageRank\n'+pr[['graph','platform','workers','iters','ms_iter','ns_edge','MEPS','speedup']].round(3).to_markdown(index=False)+'\n\n## BFS\n'+b[['graph','platform','workers','levels','ms_level','MTEPS','speedup','comm']].round(3).to_markdown(index=False)+'\n')
print(gs.to_string()); print(contr)
best=lambda df,col,g: df[df.graph==g].sort_values(col,ascending=False).iloc[0]
for g in GR:
    p=best(pr,'MEPS',g); q=best(b,'MTEPS',g); print(g,'best PR',p.cfg,round(p.MEPS,1),'| best BFS',q.cfg,round(q.MTEPS,1))

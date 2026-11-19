"""Merge other teams' (P9-P11) BFS/PageRank CSVs into one file in the common format.
usage: python3 scripts/merge_csv.py results/all_results.csv teamP9.csv teamP10.csv teamP11.csv -o results/merged.csv
Rows must carry the header below; a missing 'notes' column is filled, extra columns are rejected loudly."""
import sys, pandas as pd
H="platform,algorithm,graph,graph_type,n,m,workers,run,iterations,time_s,time_per_iter_ms,edges_processed,teps,notes".split(',')
a=sys.argv[1:]; out=a[a.index('-o')+1]; fs=[x for x in a if x!='-o' and x!=out]; dfs=[]
for f in fs:
    d=pd.read_csv(f)
    if 'notes' not in d: d['notes']=''
    miss=set(H)-set(d.columns); extra=set(d.columns)-set(H)
    if miss or extra: sys.exit(f'{f}: missing={miss} extra={extra}')
    d['source']=f; dfs.append(d[H+['source']])
pd.concat(dfs).to_csv(out,index=False); print('wrote',out,sum(map(len,dfs)),'rows')

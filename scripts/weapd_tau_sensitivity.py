# Sensitivity of the WEAPD duplicate audit to the Hamming threshold tau (all 6,877 images; MD5 + pHash <= tau, union-find)
import numpy as np, pandas as pd
d=pd.read_csv('hashes.csv'); n=len(d)
H=np.array([int(x,2) for x in d.phash],dtype=np.uint64)
LUT=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
pairs={t:[] for t in range(0,13)}
for i in range(0,n,500):
    x=H[i:i+500,None]^H[None,:]; D=LUT[x.view(np.uint8).reshape(x.shape+(8,))].sum(-1)
    ii,jj=np.nonzero(D<=12); m=jj>ii+i
    for a,b,dd in zip(ii[m]+i,jj[m],D[ii[m],jj[m]]): pairs[int(dd)].append((a,b))
md5=d.md5.values; lab=d.label.values
rows=[]
for tau in [0,2,4,6,8,10,12]:
    par=list(range(n))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    E=[p for t in range(tau+1) for p in pairs[t]]
    for a,b in E: par[f(a)]=f(b)
    for g,idx in d.groupby('md5').groups.items():
        idx=list(idx)
        for k in idx[1:]: par[f(k)]=f(idx[0])
    r=np.array([f(k) for k in range(n)]); s=pd.Series(r).value_counts(); ing=int(s[s>1].sum())
    cross=sum(lab[a]!=lab[b] for a,b in E)
    rows.append(dict(tau=tau,pairs=len(E),images_in_groups=ing,pct=round(100*ing/n,1),cross_label_pairs_pct=round(100*cross/max(len(E),1),1)))
    print(rows[-1])
pd.DataFrame(rows).to_csv('weapd_tau_sensitivity.csv',index=False)

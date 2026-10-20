# Empty-set test on EV3: (1) false alarms in distribution (probe calibrated and tested on EV3, 100 grouped/blocked re-splits),
# (2) detection of the WEAPD->EV3 shift; naive binomial vs beta-binomial (accounts for the estimated threshold) vs block-thinned version
import numpy as np, pandas as pd
from scipy.stats import binom, betabinom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
U='/mnt/user-data/uploads/anandha/results'
F=np.load(U+'/ev3_generalisation/ev3_features_fp16.npz'); bl=pd.read_csv(U+'/ev3_generalisation/ev3_blocks.csv'); Y=bl.y.values; N=len(Y)
def kq(n,a): return int(np.ceil((n+1)*(1-a)))
def tests(e,m,n,a):
    k=kq(n,a); return binom.sf(e-1,m,a), betabinom.sf(e-1,m,n+1-k,k)
def splits(groups,seed):
    fold=np.zeros(N,int)
    for f,(_,te) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=seed).split(np.zeros(N),Y,groups)): fold[te]=f
    return np.where(fold>=5)[0],np.where((fold>=3)&(fold<5))[0],np.where(fold<3)[0]
rng=np.random.default_rng(0); rows=[]
for scheme,grp,base in [('random_grouped',bl.group.values,100),('sequence_blocked',bl.block.values,200)]:
    for r in range(50):
        tr,ca,te=splits(grp,base+r)
        for key in F.files:
            X=F[key].astype(np.float32); sc=StandardScaler().fit(X[tr]); c=LogisticRegression(C=0.5,max_iter=5000).fit(sc.transform(X[tr]),Y[tr])
            pc=c.predict_proba(sc.transform(X[ca])); pt=c.predict_proba(sc.transform(X[te]))
            for a in [0.05,0.10]:
                n=len(ca); s=np.sort(1-pc[np.arange(n),Y[ca]]); q=s[min(kq(n,a),n)-1]
                emp=(pt.max(1)<1-q); m=len(te); e=int(emp.sum())
                pn,pb=tests(e,m,n,a)
                # thinning: one random test image per block
                tb=bl.block.values[te]; pick=np.array([rng.choice(np.where(tb==u)[0]) for u in np.unique(tb)])
                et=int(emp[pick].sum()); pnt,pbt=tests(et,len(pick),n,a)
                rows.append(dict(scheme=scheme,rep=r,model=key,alpha=a,n_cal=n,m=m,empty=e/m,rej_naive=pn<0.05,rej_betabin=pb<0.05,m_thin=len(pick),rej_betabin_thin=pbt<0.05))
D=pd.DataFrame(rows); D.to_csv('ev3_emptyset_false_alarms.csv',index=False)
print(D.groupby(['scheme','alpha'])[['empty','rej_naive','rej_betabin','rej_betabin_thin']].mean().round(3))
print(D.groupby(['scheme','alpha','model'])[['empty','rej_naive','rej_betabin','rej_betabin_thin']].mean().round(3))
# (2) detection of WEAPD -> EV3 shift (zero-shot sets calibrated on WEAPD validation, n = 629)
z=pd.read_csv(U+'/ev3/ev3_zero_shot_per_image.csv'); z['k']=z.file.str.split('/').str[-2:].str.join('/'); z=z.merge(bl[['k','block']],on='k')
thr={'swin_tiny_patch4_window7_224':0.835,'convnext_tiny':0.89}   # 1 - q_hat implied by the saved sets (alpha = 0.10)
for b,g in z.groupby('backbone'):
    emp=(g.top_prob.values<thr[b]); m=len(g); e=int(emp.sum()); n=629; a=0.10; k=kq(n,a)
    pb=betabinom.logsf(e-1,m,n+1-k,k)/np.log(10)
    # block-level: per block, one image; worst case over all choices = fewest empties per block
    bm=g.assign(emp=emp).groupby('block').emp.agg(['min','mean','count'])
    nb=len(bm); e_min=int(bm['min'].sum())
    pbt=betabinom.logsf(e_min-1,nb,n+1-k,k)/np.log(10)
    print(b,'empty',e,'/',m,'log10 p betabin %.1f'%pb,'| blocks',nb,'worst-case thinned empties',e_min,'log10 p %.1f'%pbt, '| blocks with mean empty >0.5:',int((bm['mean']>0.5).sum()))

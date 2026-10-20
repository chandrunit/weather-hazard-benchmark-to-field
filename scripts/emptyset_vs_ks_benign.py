# Empty-set test vs a generic label-free shift test (two-sample Kolmogorov-Smirnov on max-softmax) under a benign class-prior shift on WEAPD
import numpy as np, math, os, pandas as pd
yt=np.load('v2_seed0/test_labels.npy'); n=len(yt); cls=sorted(pd.read_csv('source_audit/meta_weapd.csv').label.unique())
EASY=[cls.index(c) for c in ['rainbow','lightning','sandstorm']]
lg=math.lgamma
def bb_sf(e,m,a,b):
    lb=lambda x,y: lg(x)+lg(y)-lg(x+y)
    ls=[lg(m+1)-lg(j+1)-lg(m-j+1)+lb(j+a,m-j+b)-lb(a,b) for j in range(e,m+1)]; mx=max(ls); return math.exp(mx)*sum(math.exp(x-mx) for x in ls)
def ks_p(x,y):
    x=np.sort(x); y=np.sort(y); z=np.concatenate([x,y]); d=np.max(np.abs(np.searchsorted(x,z,'right')/len(x)-np.searchsorted(y,z,'right')/len(y)))
    en=math.sqrt(len(x)*len(y)/(len(x)+len(y))); lam=(en+0.12+0.11/en)*d
    return max(0.0,min(1.0,2*sum((-1)**(k-1)*math.exp(-2*k*k*lam*lam) for k in range(1,101))))
def pf(e,s):
    for d in ["c3_round3","c3_round2","v2_seed0","v2_seed1","v2_seeds234"]:
        f=f"{d}/testprob_{e}_s{s}.npy"
        if os.path.exists(f): return np.load(f)
rng=np.random.default_rng(0); rows=[]
for e in ['swin_tiny_patch4_window7_224','convnext_tiny','resnet50']:
    for a in [0.10]:
        cov=[];rE=[];rK=[];cov_all=[];rE0=[];rK0=[]
        for s in range(5):
            p=pf(e,s)
            for _ in range(200):
                idx=rng.permutation(n); cal,te=idx[:n//2],idx[n//2:]; nc=len(cal); k=int(np.ceil((nc+1)*(1-a)))
                q=np.sort(1-p[cal,yt[cal]])[min(k,nc)-1]; mc=p[cal].max(1)
                for tag,sel in [('none',te),('benign',te[np.isin(yt[te],EASY)])]:
                    S=p[sel]>=1-q; E=int((S.sum(1)==0).sum()); c=S[np.arange(len(sel)),yt[sel]].mean()
                    re=bb_sf(E,len(sel),nc+1-k,k)<0.05; rk=ks_p(mc,p[sel].max(1))<0.05
                    if tag=='none': cov_all.append(c); rE0.append(re); rK0.append(rk)
                    else: cov.append(c); rE.append(re); rK.append(rk)
        rows.append(dict(model=e,no_shift_coverage=np.mean(cov_all),no_shift_alarm_empty=np.mean(rE0),no_shift_alarm_KS=np.mean(rK0),
                         benign_shift_coverage=np.mean(cov),benign_alarm_empty=np.mean(rE),benign_alarm_KS=np.mean(rK)))
        print({k:(round(v,3) if isinstance(v,float) else v) for k,v in rows[-1].items()})
pd.DataFrame(rows).to_csv('v2_final/emptyset_vs_ks_benign.csv',index=False)

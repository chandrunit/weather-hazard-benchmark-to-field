# Uncertainty and multiplicity for the EV3 field experiments
import numpy as np, pandas as pd, sys
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
U=sys.argv[1] if len(sys.argv)>1 else '/mnt/user-data/uploads/anandha/results'
rng=np.random.default_rng(0); B=2000; out=[]
def holm(p):
    p=np.asarray(p); o=np.argsort(p); a=np.empty(len(p)); run=0
    for r,i in enumerate(o): run=max(run,min(1,(len(p)-r)*p[i])); a[i]=run
    return a
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
# A. zero-shot
z=pd.read_csv(U+'/ev3/ev3_zero_shot_per_image.csv'); sp=pd.read_csv(U+'/ev3/ev3_split.csv')[['file','frozen_area_frac']]; z=z.merge(sp,on='file')
COLD=['frost','rime','glaze','snow']
for b,g in z.groupby('backbone'):
    y=g.y.values; s=g.cold_score.values; n=len(y); a=roc_auc_score(y,s)
    bs=[roc_auc_score(y[i],s[i]) for i in (rng.integers(0,n,n) for _ in range(B))]
    h=g[g.y==0]; k=h.top_class.isin(COLD).sum(); wl,wh=wilson(k,len(h))
    f=g[g.y==1]; rho,p=stats.spearmanr(f.frozen_area_frac,f.cold_score)
    rb=[stats.spearmanr(f.frozen_area_frac.values[i],f.cold_score.values[i])[0] for i in (rng.integers(0,len(f),len(f)) for _ in range(B))]
    out.append(dict(analysis='zero-shot',backbone=b,auroc=round(a,3),auroc_ci=f'{np.percentile(bs,2.5):.3f}–{np.percentile(bs,97.5):.3f}',
        healthy_flagged=f'{100*k/len(h):.1f} ({100*wl:.1f}–{100*wh:.1f})',spearman=f'{rho:.2f} ({np.percentile(rb,2.5):.2f}–{np.percentile(rb,97.5):.2f})',spearman_p=p))
# B. few-shot WEAPD vs ImageNet: two-level bootstrap over draws and test images, Holm over 2 backbones x 6 k
F=np.load(U+'/ev3_generalisation/ev3_features_fp16.npz'); bl=pd.read_csv(U+'/ev3_generalisation/ev3_blocks.csv')
Y=bl.y.values; pool=np.where(bl.split=='pool')[0]; te=np.where(bl.split=='test')[0]
fs=[]
for b in ['swin_tiny_patch4_window7_224','convnext_tiny']:
    for k in [5,10,25,50,100,'pool']:
        P={'weapd':[], 'imagenet':[]}
        nd=1 if k=='pool' else 20
        for d in range(nd):
            r2=np.random.default_rng(1000+d)
            tr=pool if k=='pool' else np.concatenate([r2.choice(pool[Y[pool]==c],k,replace=False) for c in (0,1)])
            for init in P:
                X=F[f'{b}__{init}'].astype(np.float32); sc=StandardScaler().fit(X[tr])
                P[init].append(LogisticRegression(C=0.5,max_iter=5000).fit(sc.transform(X[tr]),Y[tr]).predict_proba(sc.transform(X[te]))[:,1])
        Pw,Pi=np.array(P['weapd']),np.array(P['imagenet']); yt=Y[te]
        obs=np.mean([roc_auc_score(yt,Pw[d])-roc_auc_score(yt,Pi[d]) for d in range(nd)])
        bsd=[]
        for _ in range(1000):
            di=rng.integers(0,nd,nd); ii=rng.integers(0,len(te),len(te))
            if len(np.unique(yt[ii]))<2: continue
            bsd.append(np.mean([roc_auc_score(yt[ii],Pw[d][ii])-roc_auc_score(yt[ii],Pi[d][ii]) for d in di]))
        bsd=np.array(bsd); p=min(1,2*min((bsd<=0).mean(),(bsd>=0).mean()))
        fs.append(dict(backbone=b,k=k,diff=obs,lo=np.percentile(bsd,2.5),hi=np.percentile(bsd,97.5),p=max(p,1/len(bsd))))
fs=pd.DataFrame(fs); fs['holm']=holm(fs.p.values); fs.to_csv('fewshot_weapd_vs_imagenet_bootstrap.csv',index=False); print(fs.round(4).to_string())
# C. Grad-CAM: zero-shot vs fine-tuned eta, paired Wilcoxon per backbone x class, Holm over 4; median paired difference with bootstrap CI
gc=pd.read_csv(U+'/gradcam/gradcam_per_image.csv'); print(gc.stage.unique())
gr=[]
for b in gc.backbone.unique():
    for y in (0,1):
        a=gc[(gc.backbone==b)&(gc.y==y)&(gc.stage==gc.stage.unique()[0])].set_index('file').ratio
        c=gc[(gc.backbone==b)&(gc.y==y)&(gc.stage==gc.stage.unique()[1])].set_index('file').ratio
        d=(c-a).replace([np.inf,-np.inf],np.nan).dropna()
        p=stats.wilcoxon(d).pvalue; md=np.median(d); bm=[np.median(d.values[i]) for i in (rng.integers(0,len(d),len(d)) for _ in range(B))]
        gr.append(dict(backbone=b,y=y,n=len(d),median_diff=md,lo=np.percentile(bm,2.5),hi=np.percentile(bm,97.5),p=p))
gr=pd.DataFrame(gr); gr['holm']=holm(gr.p.values); gr.to_csv('gradcam_tests_holm.csv',index=False); print(gr.round(4).to_string())
pd.DataFrame(out).to_csv('zeroshot_ci.csv',index=False); print(pd.DataFrame(out).to_string())

import numpy as np, pandas as pd
from scipy.stats import betabinom, binom
from scipy.special import logsumexp
U='/mnt/user-data/uploads/anandha/results'
bl=pd.read_csv(U+'/ev3_generalisation/ev3_blocks.csv')
def kq(n,a): return int(np.ceil((n+1)*(1-a)))
def log10_bb_sf(e,m,n,a):
    k=kq(n,a); j=np.arange(e,m+1); return logsumexp(betabinom.logpmf(j,m,n+1-k,k))/np.log(10)
def log10_bin_sf(e,m,a):
    j=np.arange(e,m+1); return logsumexp(binom.logpmf(j,m,a))/np.log(10)
z=pd.read_csv(U+'/ev3/ev3_zero_shot_per_image.csv'); z['k']=z.file.str.split('/').str[-2:].str.join('/'); z=z.merge(bl[['k','block']],on='k')
thr={'swin_tiny_patch4_window7_224':0.835,'convnext_tiny':0.89}; rng=np.random.default_rng(0)
for b,g in z.groupby('backbone'):
    emp=(g.top_prob.values<thr[b]); m=len(g); e=int(emp.sum()); n=629; a=0.10
    blk=g.block.values; ub=np.unique(blk); ps=[]; es=[]
    for _ in range(1000):
        pick=np.array([rng.choice(np.where(blk==u)[0]) for u in ub]); et=int(emp[pick].sum()); es.append(et); ps.append(log10_bb_sf(et,len(ub),n,a))
    print(b,'e/m',e,m,'naive log10p %.0f'%log10_bin_sf(e,m,a),'betabin log10p %.0f'%log10_bb_sf(e,m,n,a),
          '| thinned (one image per block, %d blocks): empties median %d, log10 p median %.1f, max %.1f'%(len(ub),np.median(es),np.median(ps),np.max(ps)))
# WEAPD-style size check of beta-binomial vs binomial: n=315 cal, m=315 eval, alpha .1
for n,m in [(315,315),(184,271),(629,909)]:
    k=kq(n,.10); print('n',n,'m',m,'95% critical empties: binom',int(binom.isf(.05,m,.10))+1,'betabin',int(betabinom.isf(.05,m,n+1-k,k))+1)

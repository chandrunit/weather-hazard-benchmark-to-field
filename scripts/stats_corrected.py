# Corrected model comparison (paper Section 2.4, Table 6, Fig. 5, Table S1).
# Run from results/benchmark/test_probabilities/: majority vote over 5 seeds -> exact McNemar per pair,
# paired image-level bootstrap (5,000 resamples) of the accuracy difference, Holm correction over 36 pairs.
import numpy as np, glob, math, itertools, os
dirs={0:['v2_seed0','c3_round2'],1:['v2_seed1','c3_round3'],2:['v2_seeds234','c3_round3'],3:['v2_seeds234','c3_round3'],4:['v2_seeds234','c3_round3']}
y=np.load('v2_seed0/test_labels.npy')
for d in ['v2_seed1','v2_seeds234','c3_round2','c3_round3']:
    assert (np.load(d+'/test_labels.npy')==y).all(), d
names={'swin_kan_cbf_h1':'Swin-T+KAN','dino_lin_ce':'DINOv2-S','swin_tiny_patch4_window7_224':'Swin-T','vit_small_patch16_224':'ViT-S','convnext_tiny':'ConvNeXt-T','resnet50':'ResNet-50','mobilenetv3_large_100':'MobileNetV3','efficientnet_b0':'EfficientNet-B0','metecnn':'MeteCNN'}
P={}
for m in names:
    ps=[]
    for s in range(5):
        f=[d+f'/testprob_{m}_s{s}.npy' for d in dirs[s] if os.path.exists(d+f'/testprob_{m}_s{s}.npy')]
        ps.append(np.load(f[0]))
    P[m]=np.stack(ps)  # 5 x N x K
N=len(y); K=P['metecnn'].shape[2]
acc={m:(P[m].argmax(2)==y).mean(1) for m in P}
def maj(m):
    v=P[m].argmax(2); out=[]
    for i in range(N):
        c=np.bincount(v[:,i],minlength=K); top=np.flatnonzero(c==c.max())
        out.append(top[0] if len(top)==1 else top[np.argmax(P[m][:,i,top].mean(0))])
    return np.array(out)
M={m:maj(m) for m in P}
def mcn(a,b):
    ca=(M[a]==y); cb=(M[b]==y); n01=int((ca&~cb).sum()); n10=int((~ca&cb).sum()); n=n01+n10
    k=min(n01,n10); p=min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n>0 else 1.0
    return n01,n10,p
rng=np.random.default_rng(0)
def boot(a,b,B=5000):
    da=(P[a].argmax(2)==y).mean(0); db=(P[b].argmax(2)==y).mean(0); d=da-db
    idx=rng.integers(0,N,(B,N)); bs=d[idx].mean(1)
    return 100*d.mean(),100*np.percentile(bs,2.5),100*np.percentile(bs,97.5)
order=sorted(P,key=lambda m:-acc[m].mean())
print('model seedmean_acc majority_acc')
for m in order: print(f'{names[m]:16s} {100*acc[m].mean():.2f}±{100*acc[m].std(ddof=1):.2f}  maj {100*(M[m]==y).mean():.2f}')
rows=[]
for a,b in itertools.combinations(order,2):
    n01,n10,p=mcn(a,b); d,lo,hi=boot(a,b)
    rows.append([names[a],names[b],n01,n10,p,d,lo,hi])
# Holm
ps=np.array([r[4] for r in rows]); o=np.argsort(ps); m_=len(ps); adj=np.empty(m_); run=0
for r,i in enumerate(o): run=max(run,min(1,(m_-r)*ps[i])); adj[i]=run
import csv
with open('v2_final/stats_corrected.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model_a','model_b','a_only_correct','b_only_correct','mcnemar_exact_p','holm_p','acc_diff_pts','boot_ci_lo','boot_ci_hi'])
    for r,h in zip(rows,adj): w.writerow(r[:5]+[h]+r[5:])
for r,h in zip(rows,adj):
    if r[0] in ['Swin-T+KAN','DINOv2-S','Swin-T','ViT-S','ConvNeXt-T'] and r[1] in ['Swin-T+KAN','DINOv2-S','Swin-T','ViT-S','ConvNeXt-T','ResNet-50']:
        print(f'{r[0]:12s} vs {r[1]:12s} {r[2]:3d}/{r[3]:3d} p={r[4]:.3g} holm={h:.3g} diff={r[5]:+.2f} [{r[6]:+.2f},{r[7]:+.2f}]')

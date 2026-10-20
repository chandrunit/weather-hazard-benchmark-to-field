# Primary model comparison (paper Section 2.4, Table 6, Fig. 5, Table S1): seed-averaged correctness per test image,
# paired sign-flip permutation test (20,000), Holm over 36 pairs, paired bootstrap 95% CI and Bonferroni-simultaneous CI.
# Run from the results folder that contains the per-seed test-probability folders.
import numpy as np, glob, math, itertools, os, csv
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
C={m:(P[m].argmax(2)==y).mean(0) for m in P}            # seed-averaged correctness per image
order=sorted(P,key=lambda m:-acc[m].mean())
rng=np.random.default_rng(1); B=20000; npair=36
idx=rng.integers(0,N,(B,N)); S=rng.choice([-1.0,1.0],(B,N))
rows=[]
for a,b in itertools.combinations(order,2):
    d=C[a]-C[b]; D=d.mean()
    perm=(S*d).mean(1); p=(1+(np.abs(perm)>=abs(D)-1e-12).sum())/(B+1)   # paired sign-flip test
    bs=d[idx].mean(1)
    lo,hi=np.percentile(bs,[2.5,97.5]); slo,shi=np.percentile(bs,[100*0.025/npair,100*(1-0.025/npair)])
    rows.append([names[a],names[b],100*D,p,100*lo,100*hi,100*slo,100*shi])
ps=np.array([r[3] for r in rows]); o=np.argsort(ps); adj=np.empty(len(ps)); run=0
for r,i in enumerate(o): run=max(run,min(1,(len(ps)-r)*ps[i])); adj[i]=run
with open('v2_final/stats_v3_seedavg.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model_a','model_b','acc_diff_pts','perm_p','holm_p','ci95_lo','ci95_hi','simul_ci_lo','simul_ci_hi'])
    for r,h in zip(rows,adj): w.writerow([r[0],r[1],round(r[2],3),r[3],min(1,h),*[round(x,3) for x in r[4:]]])
for m in order: print(f'{names[m]:16s} seedmean {100*acc[m].mean():.2f} sd {100*acc[m].std(ddof=1):.2f}')
for r,h in zip(rows,adj): print(f'{r[0]:12s} {r[1]:16s} d={r[2]:+.2f} p={r[3]:.4f} holm={h:.4f} ci[{r[4]:+.2f},{r[5]:+.2f}] simul[{r[6]:+.2f},{r[7]:+.2f}]')

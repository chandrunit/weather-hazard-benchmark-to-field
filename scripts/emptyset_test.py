# Proposition 4 (label-free exchangeability test): empty-set rates on WEAPD and binomial tests on EV3. Run from the results folder.
# Proposition 4: under exchangeability P(C(X)=empty) <= alpha. In-distribution check on WEAPD test (200 random cal/eval halves, raw LAC sets)
import numpy as np, math, os, csv
yt=np.load('v2_seed0/test_labels.npy'); n=len(yt)
M={"MeteCNN":"metecnn","EfficientNet-B0":"efficientnet_b0","MobileNetV3-L":"mobilenetv3_large_100","ResNet-50":"resnet50","ConvNeXt-T":"convnext_tiny",
   "ViT-S/16":"vit_small_patch16_224","Swin-T":"swin_tiny_patch4_window7_224","Swin-T + KAN":"swin_kan_cbf_h1","DINOv2 ViT-S/14":"dino_lin_ce"}
def pf(e,s):
    for d in ["c3_round3","c3_round2","v2_seed0","v2_seed1","v2_seeds234"]:
        f=f"{d}/testprob_{e}_s{s}.npy"
        if os.path.exists(f): return np.load(f)
def binom_sf(k,n,p):  # P(X>=k)
    return min(1.0,sum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)) for i in range(k,n+1)))
rng=np.random.default_rng(0); rows=[]
for name,e in M.items():
    for a in [0.05,0.10]:
        emp=[];mis=[];rej=0;tot=0
        for s in range(5):
            p=pf(e,s)
            for _ in range(200):
                idx=rng.permutation(n); cal,te=idx[:n//2],idx[n//2:]
                sc=np.sort(1-p[cal,yt[cal]]); k=int(np.ceil((len(cal)+1)*(1-a))); q=sc[min(k,len(cal))-1]
                S=p[te]>=1-q; E=(S.sum(1)==0)
                emp.append(E.mean()); mis.append(1-S[np.arange(len(te)),yt[te]].mean())
                rej+= binom_sf(int(E.sum()),len(te),a)<0.05; tot+=1
        rows.append(dict(model=name,alpha=a,empty_mean=round(100*np.mean(emp),2),empty_p95=round(100*np.percentile(emp,95),2),miscov_mean=round(100*np.mean(mis),2),false_alarm_rate=round(rej/tot,3)))
        print(rows[-1])
# EV3 (zero-shot, WEAPD-calibrated sets on all 909 field images)
for b,e10,e05 in [("Swin-T",76.7,0.8),("ConvNeXt-T",84.7,1.7)]:
    for a,ep in [(0.10,e10),(0.05,e05)]:
        k=round(ep/100*909); print("EV3",b,a,k,"/909 p=",binom_sf(k,909,a))
with open('v2_final/emptyset_test_weapd.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

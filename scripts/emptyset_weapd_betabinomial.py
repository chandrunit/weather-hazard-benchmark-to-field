import numpy as np, math, os, csv
yt=np.load('v2_seed0/test_labels.npy'); n=len(yt)
M={"MeteCNN":"metecnn","EfficientNet-B0":"efficientnet_b0","MobileNetV3-L":"mobilenetv3_large_100","ResNet-50":"resnet50","ConvNeXt-T":"convnext_tiny",
   "ViT-S/16":"vit_small_patch16_224","Swin-T":"swin_tiny_patch4_window7_224","Swin-T + KAN":"swin_kan_cbf_h1","DINOv2 ViT-S/14":"dino_lin_ce"}
def pf(e,s):
    for d in ["c3_round3","c3_round2","v2_seed0","v2_seed1","v2_seeds234"]:
        f=f"{d}/testprob_{e}_s{s}.npy"
        if os.path.exists(f): return np.load(f)
lg=math.lgamma
def lbeta(a,b): return lg(a)+lg(b)-lg(a+b)
def bb_sf(e,m,a,b):
    ls=[lg(m+1)-lg(j+1)-lg(m-j+1)+lbeta(j+a,m-j+b)-lbeta(a,b) for j in range(e,m+1)]
    mx=max(ls); return math.exp(mx)*sum(math.exp(x-mx) for x in ls)
def bin_sf(e,m,p):
    return sum(math.exp(lg(m+1)-lg(i+1)-lg(m-i+1)+i*math.log(p)+(m-i)*math.log1p(-p)) for i in range(e,m+1))
rng=np.random.default_rng(0); rows=[]
for name,e in M.items():
    for a in [0.05,0.10]:
        rn=rb=tot=0; emp=[]
        for s in range(5):
            p=pf(e,s)
            for _ in range(200):
                idx=rng.permutation(n); cal,te=idx[:n//2],idx[n//2:]; nc=len(cal); k=int(np.ceil((nc+1)*(1-a)))
                sc=np.sort(1-p[cal,yt[cal]]); q=sc[min(k,nc)-1]; E=int(((p[te]>=1-q).sum(1)==0).sum()); emp.append(E/len(te))
                rn+=bin_sf(E,len(te),a)<0.05; rb+=bb_sf(E,len(te),nc+1-k,k)<0.05; tot+=1
        rows.append(dict(model=name,alpha=a,empty_mean=round(100*np.mean(emp),2),false_alarm_binomial=round(rn/tot,4),false_alarm_betabinomial=round(rb/tot,4)))
        print(rows[-1])
with open('v2_final/emptyset_test_weapd_betabin.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

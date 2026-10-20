import json, os, hashlib, numpy as np, pandas as pd
from PIL import Image
rows=[]
for s in ["train","valid","test"]:
    j=json.load(open(f"{s}/_annotations.coco.json")); cats={c["id"]:c["name"] for c in j["categories"]}
    per={}; area={}
    for a in j["annotations"]:
        n=cats[a["category_id"]]; per.setdefault(a["image_id"],set()).add(n)
        area.setdefault(a["image_id"],{}).setdefault(n,0); area[a["image_id"]][n]+=a["area"]
    for im in j["images"]:
        c=per.get(im["id"],set()); ar=area.get(im["id"],{})
        fz=sum(v for k,v in ar.items() if k.startswith("frozen")); tot=sum(ar.values()) or 1
        rows.append(dict(roboflow_split=s,file=f"{s}/{im['file_name']}",w=im["width"],h=im["height"],
            classes="|".join(sorted(c)),frost_affected=int(any(k.startswith("frozen") for k in c)),frozen_area_frac=round(fz/tot,3)))
df=pd.DataFrame(rows)
N=32; C=np.array([[np.cos(np.pi*(2*x+1)*u/(2*N)) for x in range(N)] for u in range(N)])
md5,ph,th=[],[],[]
for f in df.file:
    raw=open(f,"rb").read(); md5.append(hashlib.md5(raw).hexdigest())
    g=Image.open(f).convert("L"); a=np.asarray(g.resize((N,N),Image.LANCZOS),float)
    low=(C@a@C.T)[:8,:8].flatten(); ph.append(int("".join("1" if b else "0" for b in low>np.median(low[1:])),2))
    t=np.asarray(g.resize((48,48)),np.float32).ravel(); t-=t.mean(); th.append(t/(np.linalg.norm(t)+1e-6))
df["md5"]=md5; df["phash"]=[f"{p:016x}" for p in ph]
P=np.array(ph,dtype=np.uint64); Z=np.stack(th)
x=P[:,None]^P[None,:]; d=np.unpackbits(x.view(np.uint8).reshape(len(P),len(P),8),axis=2).sum(2)
ncc=Z@Z.T; iu=np.triu_indices(len(P),1)
pr=pd.DataFrame(dict(i=iu[0],j=iu[1],dist=d[iu],ncc=ncc[iu].round(3)))
pr=pr[pr.dist<=10]
pr["split_i"]=df.roboflow_split.values[pr.i]; pr["split_j"]=df.roboflow_split.values[pr.j]
pr["file_i"]=df.file.values[pr.i]; pr["file_j"]=df.file.values[pr.j]
print("images",len(df),"frost_affected",df.frost_affected.sum(),"healthy",(1-df.frost_affected).sum())
print("exact md5 dup copies:",df.md5.duplicated().sum())
for lo,hi in [(0,0),(1,2),(3,4),(5,6),(7,8),(9,10)]:
    b=pr[(pr.dist>=lo)&(pr.dist<=hi)]; print(f"d{lo}-{hi}: {len(b)} pairs, median ncc {b.ncc.median() if len(b) else float('nan'):.2f}, cross-split {(b.split_i!=b.split_j).sum()}")
df.to_csv("ev3_labels.csv",index=False); pr.to_csv("ev3_candidate_pairs.csv",index=False)
# contact sheet of closest pairs
s=pr.sort_values("dist").head(16); W=150
sheet=Image.new("RGB",(2*W+10,len(s)*(W+18)),"white")
from PIL import ImageDraw; dr=ImageDraw.Draw(sheet)
for k,r in enumerate(s.itertuples()):
    for jj,f in enumerate([r.file_i,r.file_j]): sheet.paste(Image.open(f).convert("RGB").resize((W,W)),(jj*(W+10),k*(W+18)))
    dr.text((0,k*(W+18)+W+2),f"d={r.dist} ncc={r.ncc:.2f} {r.split_i}|{r.split_j}",fill=(0,0,0))
sheet.save("ev3_closest_pairs.jpg",quality=85)

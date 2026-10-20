import numpy as np, pandas as pd
H=np.load('hash_weapd.npy').astype(np.uint64); M=pd.read_csv('meta_weapd.csv')
LUT=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
def dist(a,b):  # a: (n,), b: (m,) -> (n,m)
    x=(a[:,None]^b[None,:]); return LUT[x.view(np.uint8).reshape(x.shape+(8,))].sum(-1)
tr=np.where(M.split=='train')[0]; ev=np.where(M.split!='train')[0]
names=['orig','hflip','vflip','rot90','rot180','rot270','crop80','crop60']
best=np.full((len(ev),),99); how=np.array(['']*len(ev),dtype=object); nn=np.full(len(ev),-1)
for v in range(8):
    D=dist(H[ev,v],H[tr,0]); j=D.argmin(1); d=D[np.arange(len(ev)),j]
    upd=d<best; best[upd]=d[upd]; how[upd]=names[v]+'(eval)'; nn[upd]=tr[j[upd]]
for v in [6,7]:   # eval image contained in a crop of a train image
    D=dist(H[ev,0],H[tr,v]); j=D.argmin(1); d=D[np.arange(len(ev)),j]
    upd=d<best; best[upd]=d[upd]; how[upd]=names[v]+'(train)'; nn[upd]=tr[j[upd]]
R=pd.DataFrame(dict(eval_idx=ev,split=M.split.values[ev],label=M.label.values[ev],dist=best,transform=how,train_idx=nn,train_label=M.label.values[nn]))
R['same_label']=R.label==R.train_label
R.to_csv('weapd_transform_matches.csv',index=False)
for t in [4,6,8]:
    f=R[R.dist<=t]; print(f'd<={t}: eval images flagged {len(f)} (test {int((f.split=="test").sum())}, val {int((f.split=="val").sum()+(f.split=="valid").sum())}); cross-label {100*(1-f.same_label.mean()) if len(f) else 0:.1f}%'); print(f["transform"].value_counts().to_dict())
print(R.split.value_counts().to_dict())

# Source-level audit: transform-invariant perceptual hashes (orig, h/v flip, rotations, centre crops 80% and 60%) for WEAPD and EV3.
import numpy as np, pandas as pd, os, glob, sys, time
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
B=os.path.expanduser('~/mnt/anandha'); O=B+'/results/source_audit'
N=32; C=np.array([[np.cos(np.pi*(2*x+1)*u/(2*N)) for x in range(N)] for u in range(N)])
def ph(g):
    a=np.asarray(g.resize((N,N),Image.LANCZOS),dtype=np.float64); low=(C@a@C.T)[:8,:8].flatten()
    return np.packbits(low>np.median(low[1:])).view('>u8')[0]
def variants(f):
    im=Image.open(f).convert('L'); w,h=im.size; out=[]
    for t in [im, ImageOps.mirror(im), ImageOps.flip(im), im.rotate(90,expand=True), im.rotate(180), im.rotate(270,expand=True)]: out.append(ph(t))
    for s in [0.8,0.6]:
        cw,chh=int(w*s),int(h*s); l,t=(w-cw)//2,(h-chh)//2; out.append(ph(im.crop((l,t,l+cw,t+chh))))
    return out
ds=sys.argv[1]
if ds=='weapd':
    sp=pd.read_csv(B+'/results/v2_seed0/split.csv'); sp['dev']=B+'/dataset(WEAPD)/dataset/'+sp.src.str.split('/dataset/').str[-1]
    files=sp.dev.tolist(); meta=sp[['dev','label','group','split']]
else:
    files=sorted(glob.glob(B+'/data/ev3_frost_grape/*/*.jpg')); meta=pd.DataFrame(dict(dev=files))
cache=f'{O}/hash_{ds}.npy'; H=np.load(cache) if os.path.exists(cache) else np.zeros((len(files),8),dtype='>u8'); done=(H!=0).any(1)
t0=time.time()
for i,f in enumerate(files):
    if done[i]: continue
    try: H[i]=variants(f)
    except Exception as e: H[i]=1
    if time.time()-t0>150: break
np.save(cache,H); meta.to_csv(f'{O}/meta_{ds}.csv',index=False); print(ds,'done',int((H!=0).any(1).sum()),'/',len(files))

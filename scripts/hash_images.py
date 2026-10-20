# Perceptual hash (pHash, 64-bit DCT) + MD5 for every WEAPD image -> hashes.csv
import os, sys, hashlib, csv, numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
root, out = sys.argv[1], sys.argv[2]
N = 32
C = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * N)) for x in range(N)] for u in range(N)])
def phash(im):
    a = np.asarray(im.convert("L").resize((N, N), Image.LANCZOS), dtype=np.float64)
    d = C @ a @ C.T
    low = d[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return "".join("1" if b else "0" for b in bits)
import time
done = set()
if os.path.exists(out):
    done = {(r[0], r[1]) for r in csv.reader(open(out))}
t0 = time.time()
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if not done: w.writerow(["label", "file", "md5", "phash", "w", "h"])
    for c in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if (c, f) in done: continue
            if time.time() - t0 > 160: print("timeout, rerun"); sys.exit(0)
            p = os.path.join(root, c, f)
            try:
                b = open(p, "rb").read(); im = Image.open(p); im.load()
                w.writerow([c, f, hashlib.md5(b).hexdigest(), phash(im), im.size[0], im.size[1]])
            except Exception as e:
                print("bad", p, e)
        print("done", c, flush=True)

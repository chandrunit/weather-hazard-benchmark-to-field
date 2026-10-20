# Near-duplicate groups over the full original WEAPD (6,877 files): exact MD5 + pHash Hamming <= 8
import numpy as np, pandas as pd
h = pd.read_csv("hashes.csv"); n = len(h)
code = np.array([int(s, 2) for s in h.phash], dtype=np.uint64)
par = list(range(n))
def find(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
pairs = 0; cross = []
for i in range(0, n, 500):
    D = np.bitwise_count(code[i:i+500, None] ^ code[None, :])
    a, b = np.nonzero(D <= 8)
    for x, y in zip(a + i, b):
        if x < y:
            pairs += 1; par[find(x)] = find(y)
            if h.label[x] != h.label[y]: cross.append((h.label[x], h.file[x], h.label[y], h.file[y], int(D[x - i, y])))
h["group"] = [find(i) for i in range(n)]
g = h.groupby("group").size()
dup_imgs = int(g[g > 1].sum()); n_groups = int((g > 1).sum())
print(f"files {n} | near-dup pairs {pairs} | groups with >1 image {n_groups} | images in such groups {dup_imgs} ({100*dup_imgs/n:.1f}%) | redundant copies {dup_imgs-n_groups} ({100*(dup_imgs-n_groups)/n:.1f}%)")
print("largest groups:", g.sort_values(ascending=False).head(5).tolist())
print("cross-label near-dup pairs:", len(cross))
pd.DataFrame(cross, columns=["label_a","file_a","label_b","file_b","dist"]).to_csv("cross_label_pairs.csv", index=False)
per = h.assign(dup=h.group.map(g) > 1).groupby("label").dup.agg(["sum","count"]); per["pct"] = (100*per["sum"]/per["count"]).round(1)
print(per.to_string()); h[["label","file","md5","phash","group"]].to_csv("groups.csv", index=False)

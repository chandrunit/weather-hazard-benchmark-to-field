# Near-duplicate audit of the leakage-free split (seed-0 split.csv) using pHash Hamming distance
import os, csv, numpy as np, pandas as pd
from PIL import Image
R = os.path.expanduser("~/mnt/anandha/results")
DATA = os.path.expanduser("~/mnt/anandha/dataset(WEAPD)/dataset")
h = pd.read_csv(f"{R}/neardup/hashes.csv").drop_duplicates("md5")
h["code"] = h.phash.apply(lambda s: np.uint64(int(s, 2)))
sp = pd.read_csv(f"{R}/seed0/split.csv")
sp["order"] = range(len(sp))
df = sp.merge(h[["md5", "code", "file", "w", "h"]], on="md5", how="left")
print("split rows", len(sp), "matched to local images", df.code.notna().sum())
df = df[df.code.notna()].copy(); df["code"] = df.code.astype(np.uint64)
tr = df[df.split == "train"]; trc = tr.code.values.astype(np.uint64)
rows = []
for s in ["val", "test"]:
    q = df[df.split == s]
    D = np.bitwise_count(q.code.values.astype(np.uint64)[:, None] ^ trc[None, :])
    j = D.argmin(1)
    df.loc[q.index, "nn_dist"] = D.min(1)
    df.loc[q.index, "nn_label"] = tr.label.values[j]
    df.loc[q.index, "nn_file"] = tr.file.values[j]
    df.loc[q.index, "nn_trlabel"] = tr.label.values[j]
T = [0, 2, 4, 6, 8, 10, 12]
for s in ["val", "test"]:
    q = df[df.split == s]
    for t in T:
        m = q.nn_dist <= t
        rows.append(dict(split=s, max_dist=t, n=int(m.sum()), pct=round(100 * m.mean(), 2),
                         same_label=int((m & (q.nn_label == q.label)).sum()),
                         diff_label=int((m & (q.nn_label != q.label)).sum())))
aud = pd.DataFrame(rows); aud.to_csv(f"{R}/neardup/neardup_audit.csv", index=False); print(aud.to_string(index=False))
# accuracy on near-dup vs rest, per model and seed
te = df[df.split == "test"].sort_values("order")
classes = sorted(sp.label.unique()); yt = te.label.map({c: i for i, c in enumerate(classes)}).values
out = []
for seed in [0, 1]:
    for f in sorted(os.listdir(f"{R}/seed{seed}")):
        if not f.startswith("testprob_"): continue
        p = np.load(f"{R}/seed{seed}/{f}")
        full = pd.read_csv(f"{R}/seed{seed}/split.csv"); full = full[full.split == "test"].reset_index(drop=True)
        keep = full.md5.isin(te.md5).values
        pred = p.argmax(1)[keep]
        for t in [4, 8, 10]:
            m = (te.nn_dist <= t).values
            out.append(dict(model=f[9:-7], seed=seed, max_dist=t, n_near=int(m.sum()),
                            acc_near=round(100 * (pred[m] == yt[m]).mean(), 2) if m.any() else None,
                            acc_rest=round(100 * (pred[~m] == yt[~m]).mean(), 2)))
acc = pd.DataFrame(out); acc.to_csv(f"{R}/neardup/acc_near_vs_rest.csv", index=False)
print(acc[acc.max_dist == 8].to_string(index=False))
df.to_csv(f"{R}/neardup/split_with_nn.csv", index=False)
# contact sheets: sample test/train pairs at several distances
for lo, hi in [(1, 4), (5, 8), (9, 12)]:
    q = df[(df.split == "test") & df.nn_dist.between(lo, hi)].sample(frac=1, random_state=0).head(8)
    if q.empty: continue
    sheet = Image.new("RGB", (2 * 160 + 10, len(q) * 170), "white")
    for i, r in enumerate(q.itertuples()):
        a = Image.open(os.path.join(DATA, r.label, r.file)).convert("RGB").resize((160, 160))
        b = Image.open(os.path.join(DATA, r.nn_label, r.nn_file)).convert("RGB").resize((160, 160))
        sheet.paste(a, (0, i * 170)); sheet.paste(b, (170, i * 170))
    sheet.save(f"{R}/neardup/pairs_d{lo}-{hi}.jpg", quality=85)
print("sheets saved")

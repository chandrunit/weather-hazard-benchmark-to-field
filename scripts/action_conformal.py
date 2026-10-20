# N1: action-aware conformal prediction; N2: cost-sensitive hazard alerting. Uses saved test (and val) probabilities.
import os, numpy as np, pandas as pd
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(B, "action")
classes = list(pd.read_csv(f"{B}/v2_seed0/perclass_f1_mean.csv", index_col=0).columns)
ACTIONS = {"cold_protection": ["frost", "rime", "glaze", "snow"], "hail_protection": ["hail"],
           "disease_alert": ["dew", "fogsmog", "rain"], "wind_dust": ["sandstorm"], "no_action": ["rainbow", "lightning"]}
anames = list(ACTIONS); c2a = np.array([anames.index(next(a for a, cs in ACTIONS.items() if c in cs)) for c in classes])
A = np.zeros((len(classes), len(anames))); A[np.arange(len(classes)), c2a] = 1
yt = np.load(f"{B}/v2_seed0/test_labels.npy"); at = c2a[yt]
MODELS = {"MeteCNN": "metecnn", "EfficientNet-B0": "efficientnet_b0", "MobileNetV3": "mobilenetv3_large_100", "ResNet-50": "resnet50",
          "ConvNeXt-T": "convnext_tiny", "ViT-S": "vit_small_patch16_224", "Swin-T": "swin_tiny_patch4_window7_224",
          "Swin-T+KAN": "swin_kan_cbf_h1", "DINOv2-S": "dino_lin_ce"}
ENS = {"Ensemble (Swin+ViT+ConvNeXt)": ["swin_tiny_patch4_window7_224", "vit_small_patch16_224", "convnext_tiny"]}
def pf(e, s):
    for d in ["c3_round3", "c3_round2", "v2_seed0", "v2_seed1", "v2_seeds234"]:
        f = f"{B}/{d}/testprob_{e}_s{s}.npy"
        if os.path.exists(f): return np.load(f)
def probs(name, s):
    return np.mean([pf(e, s) for e in ENS[name]], 0) if name in ENS else pf(MODELS[name], s)
rng = np.random.default_rng(0); ALPHA = 0.05; REPS = 200; n = len(yt)
def conformal(p):
    out = []
    for _ in range(REPS):
        idx = rng.permutation(n); cal, te = idx[: n // 2], idx[n // 2:]
        sc = 1 - p[cal, yt[cal]]; q = np.quantile(sc, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal), method="higher")
        S = p[te] >= 1 - q; S[np.arange(len(te)), p[te].argmax(1)] = True        # never empty
        AS = (S.astype(int) @ A) > 0
        out.append(dict(cov=S[np.arange(len(te)), yt[te]].mean(), size=S.sum(1).mean(), single=(S.sum(1) == 1).mean(),
                        acov=AS[np.arange(len(te)), at[te]].mean(), asize=AS.sum(1).mean(), asingle=(AS.sum(1) == 1).mean()))
    return pd.DataFrame(out).mean()
# N2 cost: taking any protective action costs 1; leaving a hazard unprotected costs r; no_action on a hazard-free image costs 0
def cost_matrix(r):
    C = np.zeros((len(anames), len(anames)))
    for t in range(len(anames)):
        for a in range(len(anames)):
            take = 0 if anames[a] == "no_action" else 1
            miss = 0 if (a == t or anames[t] == "no_action") else r
            C[t, a] = take + miss
    return C
rows = []; crow = []
for name in list(MODELS) + list(ENS):
    for s in range(5):
        p = probs(name, s); cf = conformal(p); pa = p @ A
        acc = (p.argmax(1) == yt).mean(); aacc = (c2a[p.argmax(1)] == at).mean()
        rows.append(dict(model=name, seed=s, acc=acc, action_acc=aacc, **cf.to_dict()))
        for r in [2, 5, 10, 20]:
            C = cost_matrix(r); a_arg = c2a[p.argmax(1)]; a_bayes = (pa @ C.T).argmin(1)   # expected cost E[C(t,a)] = sum_t p_t C[t,a]
            a_bayes = (pa @ C).argmin(1)
            cost = lambda a: C[at, a].mean()
            miss_cold = lambda a: ((at == anames.index("cold_protection")) & (a != anames.index("cold_protection"))).sum() / (at == anames.index("cold_protection")).sum()
            crow.append(dict(model=name, seed=s, r=r, cost_argmax=cost(a_arg), cost_bayes=cost(a_bayes),
                             cold_miss_argmax=miss_cold(a_arg), cold_miss_bayes=miss_cold(a_bayes)))
d = pd.DataFrame(rows); d.to_csv(f"{R}/conformal_all.csv", index=False)
c = pd.DataFrame(crow); c.to_csv(f"{R}/cost_all.csv", index=False)
g = d.groupby("model", sort=False).mean(numeric_only=True).drop(columns="seed") * 100
g[["size", "asize"]] /= 100
pd.set_option("display.width", 250)
print("N1 conformal (alpha=0.05; class-level vs action-level), mean of 5 seeds x 200 splits")
print(g[["acc", "action_acc", "cov", "size", "single", "acov", "asize", "asingle"]].round(2).to_string())
h = c.groupby(["model", "r"], sort=False).mean(numeric_only=True).drop(columns="seed")
h["cost_reduction_%"] = 100 * (1 - h.cost_bayes / h.cost_argmax)
print("\nN2 expected cost per image (argmax vs Bayes-optimal) and missed cold-protection rate")
print(h.round(4).to_string())

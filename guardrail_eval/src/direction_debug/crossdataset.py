"""§15~§19 — 데이터셋 간 방향 일치, 전이, LODO, 비등방성 통제.

독립 데이터셋 3개: wildjailbreak / promptshield_test / questionset
  (promptshield_train 은 같은 데이터셋의 다른 split 이라 독립으로 세지 않음)
  (jailbreaksovertime 은 말뭉치 교란 확정 -> 별도 대조군)

AUROC < 0.5 를 자동으로 뒤집지 않는다.  부호는 과학적 결과다.
"""
import glob, itertools, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/direction_debug"
RES = ROOT / "results/direction_crossdataset"
EPS = 1e-12
CELLS = ("TP", "FP", "TN", "FN")
MAIN = ["wildjailbreak", "promptshield_test", "questionset"]
CONFOUNDED = ["jailbreaksovertime"]
EXTRA = ["promptshield_train"]
NPERM = 5000     # 지시문 3절 최소치. 메모리/시간 상한 때문에 10000 대신 5000
                 # -> 도달 가능한 최소 p = 1/5001 = 2.0e-04. 그보다 작은 p 는 주장하지 않는다.
NBOOT = 2000


def load():
    fs = sorted(glob.glob(str(ART / "cellhidden_*of*.pt")))
    ds = [torch.load(f, weights_only=False) for f in fs]
    out = {"h": torch.cat([d["h"] for d in ds]).numpy().astype(np.float64)}
    for k in ("dataset", "cell", "split_role"):
        out[k] = np.array([x for d in ds for x in d[k]])
    out["g"] = out["h"][:, 1:] - out["h"][:, :-1]
    return out


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def contrasts_from(G, cell):
    mu = {}
    for c in CELLS:
        m = cell == c
        if m.sum() == 0:
            return None
        mu[c] = G[m].mean(0)
    TP, FP, TN, FN = (mu[c] for c in CELLS)
    return {"U": TP - FP, "S": TN - FN,
            "GT": 0.5 * ((TP + FN) - (FP + TN)),
            "PRED": 0.5 * ((TP + FP) - (FN + TN)),
            "CORR": 0.5 * ((TP + TN) - (FP + FN))}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    D = load()
    L = D["g"].shape[1]
    trans = [f"L{li}->L{li+1}" for li in range(L)]
    avail = sorted(set(D["dataset"]))
    use = [d for d in MAIN + EXTRA + CONFOUNDED if d in avail]
    print(f"데이터셋: {use}")

    dirs, Gtr, Ctr = {}, {}, {}
    for ds in use:
        m = (D["dataset"] == ds) & (D["split_role"] == "train")
        Gtr[ds], Ctr[ds] = D["g"][m], D["cell"][m]
        d = contrasts_from(Gtr[ds], Ctr[ds])
        if d is None:
            print(f"  [건너뜀] {ds}: 빈 cell")
            continue
        dirs[ds] = d
        print(f"  {ds:20} train cell " + " ".join(f"{c}={int((Ctr[ds]==c).sum())}" for c in CELLS))

    # ---------- §15 데이터셋 쌍별 부호 있는 cos + 귀무 ----------
    # 치환 귀무를 **행렬곱으로 일괄** 계산한다.  cell 평균은 지시행렬 A(4,n) 와의 곱이므로
    # B개 치환을 한 번에: A_batch(4B, n) @ G_flat(n, L*H).  느린 fancy-index 루프를 없앤다.
    print(f"\n=== §15 데이터셋 쌍별 cos (치환 귀무 {NPERM}회, 최소 p = {1/(NPERM+1):.2e}) ===")
    KEYS = ("U", "S", "GT", "PRED", "CORR")
    W = {"U": {"TP": 1, "FP": -1, "TN": 0, "FN": 0},
         "S": {"TP": 0, "FP": 0, "TN": 1, "FN": -1},
         "GT": {"TP": .5, "FN": .5, "FP": -.5, "TN": -.5},
         "PRED": {"TP": .5, "FP": .5, "FN": -.5, "TN": -.5},
         "CORR": {"TP": .5, "TN": .5, "FP": -.5, "FN": -.5}}
    null = {}
    for ds in dirs:
        G32 = np.ascontiguousarray(Gtr[ds], dtype=np.float32)
        n, L_, H_ = G32.shape
        Gf = G32.reshape(n, L_ * H_)
        cnt = {c: int((Ctr[ds] == c).sum()) for c in CELLS}
        rng = np.random.default_rng(0)
        store = {k: np.empty((NPERM, L_, H_), dtype=np.float32) for k in KEYS}
        BATCH = 100
        for b0 in range(0, NPERM, BATCH):
            nb = min(BATCH, NPERM - b0)
            A = np.zeros((4 * nb, n), dtype=np.float32)
            for j in range(nb):
                perm = rng.permutation(n)
                off = 0
                for ci, c in enumerate(CELLS):
                    idx = perm[off:off + cnt[c]]; off += cnt[c]
                    A[j * 4 + ci, idx] = 1.0 / cnt[c]
                    ci_ = ci
            M = (A @ Gf).reshape(nb, 4, L_, H_)       # (nb, 4cell, L, H)
            for k in KEYS:
                acc = np.zeros((nb, L_, H_), dtype=np.float32)
                for ci, c in enumerate(CELLS):
                    w = W[k].get(c, 0)
                    if w:
                        acc += w * M[:, ci]
                store[k][b0:b0 + nb] = acc
            del A, M
        null[ds] = store
        print(f"  {ds} 귀무 {NPERM}회 생성 완료", flush=True)

    rows = []
    for a, b in itertools.combinations(sorted(dirs), 2):
        for k in ("U", "S", "GT", "PRED", "CORR"):
            cs = np.einsum("lh,lh->l", unit(dirs[a][k]), unit(dirs[b][k]))
            nc = np.einsum("blh,blh->bl", unit(null[a][k].astype(np.float64)),
                           unit(null[b][k].astype(np.float64)))
            for li in range(L):
                o = float(cs[li]); nn = nc[:, li]
                p1 = ((np.sum(nn >= o) + 1) / (NPERM + 1)) if o >= 0 else ((np.sum(nn <= o) + 1) / (NPERM + 1))
                p2 = (np.sum(np.abs(nn) >= abs(o)) + 1) / (NPERM + 1)
                rows.append({"pair": f"{a}|{b}", "effect": k, "transition": trans[li],
                             "cos": o, "null_mean": float(nn.mean()), "null_std": float(nn.std()),
                             "null_p95": float(np.percentile(nn, 95)),
                             "null_p99": float(np.percentile(nn, 99)),
                             "signed_p": float(p1), "two_sided_p": float(p2),
                             "min_attainable_p": 1 / (NPERM + 1)})
    pd.DataFrame(rows).to_csv(RES / "pairwise_direction_cosines.csv", index=False)
    pr = pd.DataFrame(rows)
    print(f"\n{'쌍':42} {'효과':5} " + " ".join(f"L{li}" for li in range(1, L)))
    for (pair, eff), g in pr[pr.effect.isin(["U", "S", "CORR"])].groupby(["pair", "effect"]):
        g = g.set_index("transition")
        print(f"{pair:42} {eff:5} " + " ".join(f"{g.loc[trans[li],'cos']:+.2f}" for li in range(1, L)))

    # ---------- §16 전이 평가 ----------
    print(f"\n=== §16 cross-dataset 전이 AUROC (부호 유지, 0.5 미만도 그대로) ===")
    tr_rows = []
    for a in dirs:
        for b in dirs:
            if a == b:
                continue
            mh = (D["dataset"] == b) & (D["split_role"] == "heldout")
            Gh, Ch = D["g"][mh], D["cell"][mh]
            if len(Ch) == 0:
                continue
            specs = [("U", np.isin(Ch, ["TP", "FP"]), (Ch == "TP")),
                     ("S", np.isin(Ch, ["TN", "FN"]), (Ch == "TN")),
                     ("CORR", np.ones(len(Ch), bool), np.isin(Ch, ["TP", "TN"]))]
            for k, mask, pos in specs:
                y = pos[mask].astype(int)
                if y.sum() < 10 or (1 - y).sum() < 10:
                    continue
                for li in range(L):
                    q = Gh[mask][:, li] @ unit(dirs[a][k][li])
                    tr_rows.append({"fit_dataset": a, "test_dataset": b, "direction": k,
                                    "transition": trans[li], "n_pos": int(y.sum()),
                                    "n_neg": int((1 - y).sum()),
                                    "auroc": float(roc_auc_score(y, q)),
                                    "auprc": float(average_precision_score(y, q))})
    td = pd.DataFrame(tr_rows)
    td.to_csv(RES / "cross_dataset_transfer.csv", index=False)
    for k in ("U", "S", "CORR"):
        sub = td[(td.direction == k) & (td.fit_dataset.isin(MAIN)) & (td.test_dataset.isin(MAIN))]
        if sub.empty:
            continue
        print(f"\n  --- delta_{k} 전이 (MAIN 3개 사이) ---")
        pv = sub.pivot_table(index=["fit_dataset", "test_dataset"], columns="transition",
                             values="auroc")
        print(pv[[trans[li] for li in range(1, L)]].round(3).to_string())

    # ---------- §17 LODO ----------
    print(f"\n=== §17 LODO: 나머지 MAIN 데이터셋의 delta_CORR 평균 방향으로 held-out 평가 ===")
    lodo = []
    mains = [d for d in MAIN if d in dirs]
    for held in mains:
        others = [d for d in mains if d != held]
        if not others:
            continue
        v = unit(np.mean([unit(dirs[o]["CORR"]) for o in others], axis=0))
        mh = (D["dataset"] == held) & (D["split_role"] == "heldout")
        Gh, Ch = D["g"][mh], D["cell"][mh]
        y = np.isin(Ch, ["TP", "TN"]).astype(int)
        rng = np.random.default_rng(0)
        for li in range(L):
            q = Gh[:, li] @ v[li]
            au = float(roc_auc_score(y, q))
            bs = []
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            for _ in range(NBOOT):
                i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
                bs.append(roc_auc_score(y[i], q[i]))
            lodo.append({"held_out": held, "trained_on": "+".join(others),
                         "transition": trans[li], "n_correct": int(y.sum()),
                         "n_incorrect": int((1 - y).sum()), "auroc": au,
                         "auprc": float(average_precision_score(y, q)),
                         "ci_lo": float(np.percentile(bs, 2.5)),
                         "ci_hi": float(np.percentile(bs, 97.5))})
    ld = pd.DataFrame(lodo)
    ld.to_csv(RES / "lodo_correctness.csv", index=False)
    if not ld.empty:
        print(ld.pivot_table(index="held_out", columns="transition", values="auroc")
              [[trans[li] for li in range(1, L)]].round(3).to_string())

    # ---------- §19 whitening 통제 ----------
    print(f"\n=== §19 비등방성 통제: 원본 cos vs 화이트닝 cos (진단용, MAIN 결과 대체 아님) ===")
    wr = []
    for a, b in itertools.combinations(mains, 2):
        for k in ("U", "S", "CORR"):
            for li in range(L):
                raw = float(np.dot(unit(dirs[a][k][li]), unit(dirs[b][k][li])))
                # 화이트닝 행렬은 각 데이터셋 TRAIN 에서만 적합 (정칙화)
                ws = []
                for src in (a, b):
                    X = Gtr[src][:, li]
                    X = X - X.mean(0)
                    C = X.T @ X / max(len(X) - 1, 1)
                    C += np.trace(C) / C.shape[0] * 0.1 * np.eye(C.shape[0])   # ridge
                    ws.append(np.linalg.inv(np.linalg.cholesky(C)))
                va = ws[0] @ dirs[a][k][li]; vb = ws[1] @ dirs[b][k][li]
                wc = float(np.dot(va / (np.linalg.norm(va) + EPS), vb / (np.linalg.norm(vb) + EPS)))
                wr.append({"pair": f"{a}|{b}", "effect": k, "transition": trans[li],
                           "cos_raw": raw, "cos_whitened": wc})
    wd = pd.DataFrame(wr)
    wd.to_csv(RES / "whitened_cosines.csv", index=False)
    for k in ("CORR",):
        s = wd[wd.effect == k]
        print(f"  delta_{k}: 원본 평균 {s.cos_raw.mean():+.3f}  화이트닝 평균 {s.cos_whitened.mean():+.3f}")
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main()

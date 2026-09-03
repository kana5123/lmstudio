"""§13 split 재현성 · §14 cross-dataset 재현성 · §15 치환 귀무.

치환 설계(§15) 명시:
  **보존하는 것**: 정답 라벨(attack/benign) 층화, 네 cell 의 개수, 표현 자체.
  **깨뜨리는 것**: 정오(correct/incorrect) 배정과 표현 사이의 연결.
  구현: TRAIN 안에서만, attack 표본 사이에서 TP/FN 배정을 섞고 benign 표본 사이에서
        TN/FP 배정을 섞는다.  그렇게 만든 가짜 delta_CORR 로 **진짜 라벨의 held-out**
        correctness AUROC 를 계산해 귀무분포를 만든다.
  이론적 등방 귀무(1/sqrt(d))는 쓰지 않는다 — 표현이 비등방이기 때문(유효랭크 1.2~11.4 실측).
"""
import argparse, itertools, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direction_correctness.correctness_direction import (
    load, unit, group_split, cell_means, directions, auroc, perm_labels,
    CELLS, CORRECT, ATTACK, SEEDS, EPS, MIN_CELL_TEST)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/direction_correctness"
NPERM = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", default="move", choices=["move", "raw"])
    ap.add_argument("--nperm", type=int, default=NPERM)
    a = ap.parse_args()
    D = load()
    X_all = D["g"] if a.repr == "move" else D["h"]
    L = X_all.shape[1]
    names = ([f"L{li}->L{li+1}" for li in range(L)] if a.repr == "move"
             else [f"L{li}" for li in range(L)])
    tag = a.repr

    per = {}          # (ds, seed) -> dict of directions
    split = {}        # (ds, seed) -> (tr, te)
    data = {}
    for ds in sorted(set(D["dataset"])):
        m = D["dataset"] == ds
        Xd, cd, dupd = X_all[m], D["cell"][m], D["dup"][m]
        tot = {c: int((cd == c).sum()) for c in CELLS}
        if min(tot.values()) < 40:
            continue
        data[ds] = (Xd, cd, dupd)
        for seed in SEEDS:
            tr, te = group_split(dupd, cd, seed)
            nte = {c: int((cd[te] == c).sum()) for c in CELLS}
            if min(nte.values()) < MIN_CELL_TEST:
                continue
            d = directions(cell_means(Xd[tr], cd[tr]))
            per[(ds, seed)] = d
            split[(ds, seed)] = (tr, te)
    dss = sorted({k[0] for k in per})
    print(f"표현={tag}  데이터셋 {dss}")

    # ---------- §13 split 재현성 ----------
    rows = []
    for ds in dss:
        sds = sorted(s for (d0, s) in per if d0 == ds)
        for k in ("CORR", "LABEL", "PRED", "d_danger", "d_safe"):
            for li in range(L):
                cs = [float(np.dot(unit(per[(ds, s1)][k][li]), unit(per[(ds, s2)][k][li])))
                      for s1, s2 in itertools.combinations(sds, 2)]
                rows.append({"dataset": ds, "repr": tag, "direction": k, "layer": names[li],
                             "n_pairs": len(cs), "mean": float(np.mean(cs)),
                             "std": float(np.std(cs)), "min": float(np.min(cs)),
                             "max": float(np.max(cs))})
    sr = pd.DataFrame(rows)
    sr.to_csv(RES / f"split_reproducibility_by_layer__{tag}.csv", index=False)
    print("\n=== §13 seed 간 방향 cos 평균 (delta_CORR) ===")
    p = sr[sr.direction == "CORR"].pivot_table(index="dataset", columns="layer", values="mean")
    print(p[[n for n in names]].round(3).to_string())

    # ---------- §14 cross-dataset ----------
    rows = []
    for a1, b1 in itertools.combinations(dss, 2):
        for k in ("CORR", "LABEL", "PRED"):
            for li in range(L):
                va = unit(np.mean([unit(per[(a1, s)][k][li]) for s in SEEDS if (a1, s) in per], 0))
                vb = unit(np.mean([unit(per[(b1, s)][k][li]) for s in SEEDS if (b1, s) in per], 0))
                rows.append({"pair": f"{a1}|{b1}", "repr": tag, "direction": k,
                             "layer": names[li], "cos": float(np.dot(va, vb))})
    # 전이 AUROC (A 의 방향 -> B 의 held-out)
    tr_rows = []
    for a1 in dss:
        va_all = {k: unit(np.mean([unit(per[(a1, s)][k]) for s in SEEDS if (a1, s) in per], 0))
                  for k in ("CORR", "LABEL")}
        for b1 in dss:
            if a1 == b1:
                continue
            Xd, cd, dupd = data[b1]
            for seed in SEEDS:
                if (b1, seed) not in split:
                    continue
                _, te = split[(b1, seed)]
                cte = cd[te]
                y_c = np.isin(cte, CORRECT).astype(int)
                y_l = np.isin(cte, ATTACK).astype(int)
                for li in range(L):
                    s = Xd[te][:, li] @ va_all["CORR"][li]
                    sl = Xd[te][:, li] @ va_all["LABEL"][li]
                    tr_rows.append({"fit_dataset": a1, "test_dataset": b1, "repr": tag,
                                    "seed": seed, "layer": names[li],
                                    "CORR_auroc_correctness": auroc(y_c, s),
                                    "CORR_auroc_label": auroc(y_l, s),
                                    "LABEL_auroc_label": auroc(y_l, sl),
                                    "LABEL_auroc_correctness": auroc(y_c, sl)})
    pd.DataFrame(rows + []).to_csv(RES / f"cross_dataset_reproducibility_by_layer__{tag}.csv",
                                   index=False)
    td = pd.DataFrame(tr_rows)
    td.to_csv(RES / f"cross_dataset_transfer__{tag}.csv", index=False)
    cd_ = pd.DataFrame(rows)
    print("\n=== §14 데이터셋 간 delta_CORR cos (seed 평균 방향) ===")
    print(cd_[cd_.direction == "CORR"].pivot_table(index="pair", columns="layer",
                                                   values="cos")[[n for n in names]].round(3).to_string())
    if not td.empty:
        print("\n=== §14 cross-dataset 전이 correctness AUROC (delta_CORR, seed 평균) ===")
        print(td.pivot_table(index=["fit_dataset", "test_dataset"], columns="layer",
                             values="CORR_auroc_correctness")[[n for n in names]].round(3).to_string())

    # ---------- §15 치환 귀무 ----------
    rows = []
    print(f"\n=== §15 치환 귀무 {a.nperm}회 (최소 도달 p = {1/(a.nperm+1):.2e}) ===")
    for ds in dss:
        Xd, cd, dupd = data[ds]
        for seed in SEEDS[:2]:                 # 시간 절약: seed 2개로 귀무 추정
            if (ds, seed) not in split:
                continue
            tr, te = split[(ds, seed)]
            ctr, cte = cd[tr], cd[te]
            y_c = np.isin(cte, CORRECT).astype(int)
            obs = [auroc(y_c, Xd[te][:, li] @ unit(per[(ds, seed)]["CORR"][li])) for li in range(L)]
            rng = np.random.default_rng(1000 + seed)
            null = np.empty((a.nperm, L))
            Xtr, Xte = Xd[tr], Xd[te]
            for b in range(a.nperm):
                cp = perm_labels(ctr, rng)
                dp = directions(cell_means(Xtr, cp))
                for li in range(L):
                    null[b, li] = auroc(y_c, Xte[:, li] @ unit(dp["CORR"][li]))
            for li in range(L):
                nn = null[:, li]
                pv = (np.sum(nn >= obs[li]) + 1) / (a.nperm + 1)
                rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": names[li],
                             "observed_auroc": obs[li], "null_mean": float(nn.mean()),
                             "null_std": float(nn.std()),
                             "null_p95": float(np.percentile(nn, 95)),
                             "empirical_p": float(pv), "n_perm": a.nperm,
                             "min_attainable_p": 1 / (a.nperm + 1)})
            print(f"  {ds} seed{seed}: 최고 관측 {max(obs):.3f}, "
                  f"귀무평균 {null.mean():.3f}, 최소 p {min(r['empirical_p'] for r in rows[-L:]):.4f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(RES / f"permutation_null_by_layer__{tag}.csv", index=False)
    print(f"\n저장 -> {RES}/*__{tag}.csv")


if __name__ == "__main__":
    main()

"""label 효과를 상쇄한 correctness main-effect 방향의 held-out 검증.

기존 실험(src/direction_debug/four_cell.py, results/direction_debug/*)은 **덮어쓰지 않는다.**

표현(representation) — 기존과 동일 조건 유지:
  기존 four_cell.py 는 `g^(l) = h^(l) - h^(l-1)` (층간 **이동벡터**)를 썼다.
  지시문 8절 표기는 `h_i^(l)` 이라 모호하므로 **둘 다** 돌리고 둘 다 보고한다.
    repr="move" : g^(l) = h^(l) - h^(l-1)   <- 기존과 동일. PRIMARY
    repr="raw"  : h^(l)                     <- 지시문 표기 그대로. SECONDARY
  pooling 은 기존과 동일하게 CLS 위치(색인 0), best_window(=UNSAFE 확률 최대 창).
  방향 계산 전 정규화는 하지 않고, 사영 직전에만 단위벡터로 만든다(지시문 7절).

2x2 요인 설계 (지시문 17절):
    요인1 Y = 정답 라벨   attack=+1 / benign=-1
    요인2 C = 정오        correct=+1 / incorrect=-1
    TP=(attack,correct) FP=(benign,incorrect) FN=(attack,incorrect) TN=(benign,correct)

  LABEL 주효과       = [(TP+FN) - (FP+TN)] / 2
  CORRECTNESS 주효과 = [(TP+TN) - (FP+FN)] / 2      <- 우리가 찾는 것 (beta_C 에 대응)
  LABEL x CORR 상호작용 = [(TP+FP) - (FN+TN)] / 2   ~ 예측(predicted attack - benign)

  effect-coded 선형모형 h = b0 + bL*L + bC*C + bLC*(L*C) + e 에서,
  cell 당 동일 가중을 주면 balanced cell-mean contrast 가 정확히 2*beta 가 된다.
  (cell 이 균형 가중이면 설계행렬이 직교하므로 각 대비가 해당 계수를 독립적으로 준다.)
"""
import argparse, glob, itertools, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/direction_debug"
RES = ROOT / "results/direction_correctness"
OUTA = ROOT / "artifacts/direction_correctness"
CELLS = ("TP", "FP", "TN", "FN")
CORRECT = ("TP", "TN")
ATTACK = ("TP", "FN")
PRED_ATK = ("TP", "FP")
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.30
NPERM = 500
NBOOT = 1000
EPS = 1e-12
MIN_CELL_TEST = 15          # 검정 가능한 최소 test cell 크기 (heuristic, 이론값 아님)


def load():
    fs = sorted(glob.glob(str(ART / "cellhidden_*of*.pt")))
    assert fs, "cell 은닉표현 없음 — artifacts/direction_debug/cellhidden_*.pt"
    ds = [torch.load(f, weights_only=False) for f in fs]
    h = torch.cat([d["h"] for d in ds]).numpy().astype(np.float64)   # (n, L+1, H)
    out = {"h": h, "g": h[:, 1:] - h[:, :-1]}
    for k in ("dataset", "cell", "dup", "sample_id", "split_role"):
        out[k] = np.array([x for d in ds for x in d[k]])
    return out


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def group_split(dup, cell, seed, test_frac=TEST_FRAC):
    """중복그룹(dup)을 쪼개지 않으면서 cell 비율을 보존하는 분할.
    같은 dup 은 항상 같은 쪽으로 간다(누수 방지)."""
    rng = np.random.default_rng(seed)
    # 그룹의 대표 cell = 그 그룹 첫 행의 cell
    first = {}
    for i, d in enumerate(dup):
        if d not in first:
            first[d] = cell[i]
    groups = np.array(list(first.keys()))
    strat = np.array([first[g] for g in groups])
    test_groups = set()
    for c in CELLS:
        gc = groups[strat == c]
        if len(gc) == 0:
            continue
        k = int(round(len(gc) * test_frac))
        test_groups |= set(rng.permutation(gc)[:k].tolist())
    is_test = np.array([d in test_groups for d in dup])
    return ~is_test, is_test


def cell_means(X, cell):
    """X:(n, L, H) -> dict cell->(L,H).  cell 이 비면 None."""
    mu = {}
    for c in CELLS:
        m = cell == c
        mu[c] = X[m].mean(0) if m.sum() > 0 else None
    return mu


def directions(mu):
    """지시문 4~6절.  전부 balanced cell-mean contrast."""
    if any(mu[c] is None for c in CELLS):
        return None
    TP, FP, TN, FN = mu["TP"], mu["FP"], mu["TN"], mu["FN"]
    d = {
        "d_danger": TP - FP,                        # A. 기존 predicted-danger 대비
        "d_safe": TN - FN,                          # B. predicted-safe 대비
        "CORR": 0.5 * ((TP + TN) - (FP + FN)),      # C. correctness 주효과
        "LABEL": 0.5 * ((TP + FN) - (FP + TN)),     # 5. label 주효과 (attack - benign)
        "PRED": 0.5 * ((TP + FP) - (FN + TN)),      # 6. 상호작용 ~ 예측효과
    }
    # ---- 지시문 22절 구현 검증 (assert) ----
    a = (d["d_danger"] + d["d_safe"]) / 2
    b = (d["d_danger"] - d["d_safe"]) / 2
    assert np.abs(a - d["CORR"]).max() < 1e-9, np.abs(a - d["CORR"]).max()
    assert np.abs(b - d["LABEL"]).max() < 1e-9, np.abs(b - d["LABEL"]).max()
    return d


def auroc(y, s):
    """부호를 절대 뒤집지 않는다.  한 클래스뿐이면 nan."""
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def boot_ci(y, s, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 3 or len(neg) < 3:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        out.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_labels(cell, rng):
    """지시문 15절 치환 설계.
    **보존**: 정답 라벨(attack/benign) 층화, 네 cell 의 개수.
    **파괴**: 정오(correct/incorrect)와 표현 사이의 연결.
    구현: attack 표본 안에서 TP/FN 배정을 섞고, benign 표본 안에서 TN/FP 배정을 섞는다.
    """
    out = np.empty_like(cell)
    atk = np.isin(cell, ATTACK)
    ia = np.flatnonzero(atk); ib = np.flatnonzero(~atk)
    n_tp = int((cell == "TP").sum()); n_tn = int((cell == "TN").sum())
    pa = rng.permutation(ia); pb = rng.permutation(ib)
    out[pa[:n_tp]] = "TP"; out[pa[n_tp:]] = "FN"
    out[pb[:n_tn]] = "TN"; out[pb[n_tn:]] = "FP"
    return out


def evaluate(dirs, Xte, cte, layer_names):
    """held-out 사영 평가.  AUROC 부호 유지."""
    y_corr = np.isin(cte, CORRECT).astype(int)
    y_lab = np.isin(cte, ATTACK).astype(int)
    m_dan = np.isin(cte, PRED_ATK)          # 예측=attack 부분집합 (TP/FP)
    m_saf = ~m_dan                          # 예측=benign 부분집합 (TN/FN)
    rows = []
    for li, ln in enumerate(layer_names):
        r = {"transition": ln}
        for k in ("CORR", "LABEL", "PRED", "d_danger", "d_safe"):
            v = unit(dirs[k][li])
            s = Xte[:, li] @ v
            r[f"{k}__auroc_correctness"] = auroc(y_corr, s)
            r[f"{k}__auroc_label"] = auroc(y_lab, s)
            if m_dan.sum() > 0:
                r[f"{k}__auroc_danger_TPvFP"] = auroc((cte[m_dan] == "TP").astype(int), s[m_dan])
            if m_saf.sum() > 0:
                r[f"{k}__auroc_safe_TNvFN"] = auroc((cte[m_saf] == "TN").astype(int), s[m_saf])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", default="move", choices=["move", "raw"])
    ap.add_argument("--nperm", type=int, default=NPERM)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); OUTA.mkdir(parents=True, exist_ok=True)
    D = load()
    X_all = D["g"] if a.repr == "move" else D["h"]
    L = X_all.shape[1]
    names = ([f"L{li}->L{li+1}" for li in range(L)] if a.repr == "move"
             else [f"L{li}" for li in range(L)])
    tag = a.repr
    print(f"표현={tag}  모양={X_all.shape}  층/전이 {L}개")

    datasets = sorted(set(D["dataset"]))
    eff_rows, auc_rows, sub_rows, leak_rows, align_rows, cnt_rows = [], [], [], [], [], []
    saved_dirs = {}

    for ds in datasets:
        m = D["dataset"] == ds
        Xd, cd, dupd = X_all[m], D["cell"][m], D["dup"][m]
        tot = {c: int((cd == c).sum()) for c in CELLS}
        if min(tot.values()) < 40:
            print(f"  [건너뜀] {ds}: cell 최소 {min(tot.values())} < 40  {tot}")
            continue
        print(f"\n=== {ds}  전체 cell {tot} ===")
        for seed in SEEDS:
            tr, te = group_split(dupd, cd, seed)
            ctr, cte = cd[tr], cd[te]
            ntr = {c: int((ctr == c).sum()) for c in CELLS}
            nte = {c: int((cte == c).sum()) for c in CELLS}
            if min(nte.values()) < MIN_CELL_TEST or min(ntr.values()) < 10:
                print(f"  seed{seed}: test cell 부족 {nte} -> 건너뜀")
                continue
            cnt_rows.append({"dataset": ds, "repr": tag, "seed": seed,
                             **{f"train_n_{c}": ntr[c] for c in CELLS},
                             **{f"test_n_{c}": nte[c] for c in CELLS}})
            mu = cell_means(Xd[tr], ctr)
            dirs = directions(mu)
            saved_dirs[(ds, seed)] = {k: v.copy() for k, v in dirs.items()}

            for li, ln in enumerate(names):
                eff_rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": ln,
                                 **{f"norm_{k}": float(np.linalg.norm(dirs[k][li])) for k in dirs},
                                 **{f"n_{c}": ntr[c] for c in CELLS}})

            ev = evaluate(dirs, Xd[te], cte, names)
            y_corr = np.isin(cte, CORRECT).astype(int)
            for li, r in enumerate(ev):
                v = unit(dirs["CORR"][li]); s = Xd[te][:, li] @ v
                lo, hi = boot_ci(y_corr, s, seed=seed)
                auc_rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": r["transition"],
                                 "auroc_correctness": r["CORR__auroc_correctness"],
                                 "ci_lo": lo, "ci_hi": hi,
                                 **{f"n_{c}": nte[c] for c in CELLS}})
                sub_rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": r["transition"],
                                 "auroc_danger_TPvFP": r.get("CORR__auroc_danger_TPvFP"),
                                 "auroc_safe_TNvFN": r.get("CORR__auroc_safe_TNvFN"),
                                 "n_TP": nte["TP"], "n_FP": nte["FP"],
                                 "n_TN": nte["TN"], "n_FN": nte["FN"]})
                leak_rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": r["transition"],
                                  "CORR_auroc_correctness": r["CORR__auroc_correctness"],
                                  "CORR_auroc_label": r["CORR__auroc_label"],
                                  "LABEL_auroc_label": r["LABEL__auroc_label"],
                                  "LABEL_auroc_correctness": r["LABEL__auroc_correctness"],
                                  "PRED_auroc_correctness": r["PRED__auroc_correctness"],
                                  "d_danger_auroc_danger": r.get("d_danger__auroc_danger_TPvFP"),
                                  "d_safe_auroc_safe": r.get("d_safe__auroc_safe_TNvFN")})
                align_rows.append({"dataset": ds, "repr": tag, "seed": seed, "layer": r["transition"],
                                   "cos_CORR_LABEL": float(np.dot(unit(dirs["CORR"][li]), unit(dirs["LABEL"][li]))),
                                   "cos_CORR_PRED": float(np.dot(unit(dirs["CORR"][li]), unit(dirs["PRED"][li]))),
                                   "cos_LABEL_PRED": float(np.dot(unit(dirs["LABEL"][li]), unit(dirs["PRED"][li]))),
                                   "cos_danger_safe": float(np.dot(unit(dirs["d_danger"][li]), unit(dirs["d_safe"][li]))),
                                   "cos_danger_LABEL": float(np.dot(unit(dirs["d_danger"][li]), unit(dirs["LABEL"][li]))),
                                   "cos_safe_LABEL": float(np.dot(unit(dirs["d_safe"][li]), unit(dirs["LABEL"][li])))})
            best = max(range(L), key=lambda i: auc_rows[-L + i]["auroc_correctness"])
            print(f"  seed{seed}  train {ntr}  test {nte}  "
                  f"최고 correctness AUROC {auc_rows[-L+best]['auroc_correctness']:.3f} @ {names[best]}")

    for nm, rows in (("effect_directions_by_layer", eff_rows),
                     ("correctness_auroc_by_layer", auc_rows),
                     ("subgroup_auroc_by_layer", sub_rows),
                     ("label_leakage_by_layer", leak_rows),
                     ("direction_alignment_by_layer", align_rows),
                     ("cell_counts_by_split", cnt_rows)):
        pd.DataFrame(rows).to_csv(RES / f"{nm}__{tag}.csv", index=False)
    np.savez_compressed(OUTA / f"directions__{tag}.npz",
                        **{f"{ds}|s{sd}|{k}": v for (ds, sd), d in saved_dirs.items()
                           for k, v in d.items()})
    print(f"\n저장 -> {RES}/*__{tag}.csv, {OUTA}/directions__{tag}.npz")


if __name__ == "__main__":
    main()

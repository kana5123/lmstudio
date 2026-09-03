"""분석 A~D — centroid / regularized linear / shrinkage LDA / branch-specific.

A (B0 기준선): Delta_CORR = 1/2[(mu_TP+mu_TN) - (mu_FP+mu_FN)],  s = unit(Delta)^T x
B: L2 로지스틱 회귀.  cell 당 동일 총가중.  규제강도 C 는 **TRAIN 내부 검증**에서만 선택.
C: Ledoit-Wolf 축소 LDA (고차원 공분산 문제 회피).
D: 가지별 probe — Probe-U(예측 UNSAFE 안에서 TP vs FP), Probe-S(예측 SAFE 안에서 TN vs FN).

용량 통제(지시문 33절): 모든 probe 를 라벨 섞기 버전으로도 학습해 같이 저장한다.
층 선택 편향 방지(34절): 전 층 결과를 전부 보존한다. 최고 층만 고르지 않는다.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, boot_delta_ci, wcsv, CELLS, CORRECT, PRED_UNSAFE,
    ALL_ANALYSED, SEEDS, INNER_VAL_FRAC, RES, EPS)

C_GRID = [0.01, 0.1, 1.0]
MAIN_REPR = ["R1_h", "R2_g"]
SEC_REPR = ["R3_h_norm", "R4_g_norm", "R5_cat_hh", "R6_cat_hg"]


def centroid_dir(X, cell):
    mu = {c: X[cell == c].mean(0) for c in CELLS}
    d = 0.5 * ((mu["TP"] + mu["TN"]) - (mu["FP"] + mu["FN"]))
    return d / (np.linalg.norm(d) + EPS)


def fit_logreg(Xtr, ytr, wtr, seed, Cs=C_GRID):
    """C 를 TRAIN 내부 분할에서만 고른다.  TEST 를 보지 않는다."""
    rng = np.random.default_rng(seed)
    n = len(ytr); idx = rng.permutation(n); k = int(n * INNER_VAL_FRAC)
    iv, it = idx[:k], idx[k:]
    sc = StandardScaler().fit(Xtr[it])
    best, bestC = -np.inf, Cs[0]
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=300, solver="lbfgs")
        m.fit(sc.transform(Xtr[it]), ytr[it], sample_weight=wtr[it])
        a = auroc(ytr[iv], m.predict_proba(sc.transform(Xtr[iv]))[:, 1])
        if not np.isnan(a) and a > best:
            best, bestC = a, C
    sc2 = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=bestC, max_iter=300, solver="lbfgs")
    m.fit(sc2.transform(Xtr), ytr, sample_weight=wtr)
    return m, sc2, bestC, best


def run(dsets, reprs, seeds, tag):
    D = load(); R = representations(D["h"])
    rows_c, rows_l, rows_d, rows_b = [], [], [], []
    t0 = time.time()
    for ds in dsets:
        dm = D["dataset"] == ds
        cell_d, dup_d = D["cell"][dm], D["dup"][dm]
        for rname in reprs:
            Xall, lnames = R[rname]
            Xd = Xall[dm]
            for seed in seeds:
                tr, te = group_split(dup_d, cell_d, seed)
                ctr, cte = cell_d[tr], cell_d[te]
                ytr, yte = y_error(ctr), y_error(cte)
                wtr = cell_weights(ctr)
                mdan_te = np.isin(cte, PRED_UNSAFE); msaf_te = ~mdan_te
                mdan_tr = np.isin(ctr, PRED_UNSAFE); msaf_tr = ~mdan_tr
                for li, ln in enumerate(lnames):
                    Xtr, Xte = Xd[tr][:, li].astype(np.float64), Xd[te][:, li].astype(np.float64)
                    base = dict(dataset=ds, repr=rname, seed=seed, layer=ln,
                                n_train=int(tr.sum()), n_test=int(te.sum()),
                                **{f"test_n_{c}": int((cte == c).sum()) for c in CELLS})

                    # ---- A. centroid (correct - incorrect 방향, 부호 유지) ----
                    v = centroid_dir(Xtr, ctr)
                    s_c = Xte @ v
                    lo, hi = boot_ci(1 - yte, s_c, seed=seed)      # 양성 = correct
                    rows_c.append({**base, "auroc": auroc(1 - yte, s_c),
                                   "auprc": auprc(1 - yte, s_c), "ci_lo": lo, "ci_hi": hi,
                                   "norm_delta": float(np.linalg.norm(
                                       0.5 * ((Xtr[ctr == "TP"].mean(0) + Xtr[ctr == "TN"].mean(0))
                                              - (Xtr[ctr == "FP"].mean(0) + Xtr[ctr == "FN"].mean(0)))))})

                    # ---- B. 규제 로지스틱 (양성 = incorrect) ----
                    m, sc, bestC, ival = fit_logreg(Xtr, ytr, wtr, seed)
                    s_tr = m.decision_function(sc.transform(Xtr))
                    s_te = m.decision_function(sc.transform(Xte))
                    lo, hi = boot_ci(yte, s_te, seed=seed)
                    # 라벨 섞기 용량 통제
                    rng = np.random.default_rng(1000 + seed)
                    ysh = rng.permutation(ytr)
                    msh = LogisticRegression(C=bestC, max_iter=300, solver="lbfgs")
                    msh.fit(sc.transform(Xtr), ysh, sample_weight=wtr)
                    a_sh = auroc(yte, msh.decision_function(sc.transform(Xte)))
                    # centroid 대비 개선폭 (centroid 는 correct 양성이므로 부호 맞춰 -s_c 사용)
                    dlo, dhi = boot_delta_ci(yte, -s_c, s_te, seed=seed)
                    rows_l.append({**base, "auroc_train": auroc(ytr, s_tr),
                                   "auroc_innerval": ival, "auroc": auroc(yte, s_te),
                                   "auprc": auprc(yte, s_te), "ci_lo": lo, "ci_hi": hi,
                                   "C": bestC, "coef_norm": float(np.linalg.norm(m.coef_)),
                                   "auroc_shuffled_label": a_sh,
                                   "auroc_centroid_as_error": auroc(yte, -s_c),
                                   "delta_vs_centroid": auroc(yte, s_te) - auroc(yte, -s_c),
                                   "delta_ci_lo": dlo, "delta_ci_hi": dhi})

                    # ---- C. 축소 LDA ----
                    try:
                        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                        lda.fit(sc.transform(Xtr), ytr)
                        s_lda = lda.decision_function(sc.transform(Xte))
                        lo2, hi2 = boot_ci(yte, s_lda, seed=seed)
                        rows_d.append({**base, "auroc": auroc(yte, s_lda),
                                       "auprc": auprc(yte, s_lda), "ci_lo": lo2, "ci_hi": hi2,
                                       "coef_norm": float(np.linalg.norm(lda.coef_))})
                    except Exception as e:
                        rows_d.append({**base, "auroc": np.nan, "auprc": np.nan,
                                       "ci_lo": np.nan, "ci_hi": np.nan, "coef_norm": np.nan,
                                       "error": str(e)[:80]})

                    # ---- D. 가지별 probe ----
                    br = {**base}
                    for bn, mtr_, mte_, pos in (("U", mdan_tr, mdan_te, "TP"),
                                                ("S", msaf_tr, msaf_te, "TN")):
                        ytr_b = (ctr[mtr_] != pos).astype(int)    # 1 = incorrect
                        yte_b = (cte[mte_] != pos).astype(int)
                        if min(ytr_b.sum(), (1 - ytr_b).sum()) < 10 or \
                           min(yte_b.sum(), (1 - yte_b).sum()) < 5:
                            br[f"auroc_{bn}"] = np.nan; br[f"n_{bn}_test"] = int(mte_.sum())
                            continue
                        wb = cell_weights(ctr[mtr_])
                        mb, scb, Cb, _ = fit_logreg(Xtr[mtr_], ytr_b, wb, seed)
                        sb = mb.decision_function(scb.transform(Xte[mte_]))
                        br[f"auroc_{bn}"] = auroc(yte_b, sb)
                        br[f"n_{bn}_test"] = int(mte_.sum()); br[f"C_{bn}"] = Cb
                    rows_b.append(br)
                print(f"  {ds:20} {rname:10} seed{seed} 완료 ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / f"centroid_baseline{tag}.csv", rows_c)
    wcsv(RES / f"logistic_probe{tag}.csv", rows_l)
    wcsv(RES / f"lda_probe{tag}.csv", rows_d)
    wcsv(RES / f"branch_specific_probes{tag}.csv", rows_b)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="main", choices=["main", "secondary"])
    a = ap.parse_args()
    if a.which == "main":
        run(ALL_ANALYSED, MAIN_REPR, SEEDS, "")
    else:
        run(["wildjailbreak", "promptshield_test"], SEC_REPR, [0], "_secondary")

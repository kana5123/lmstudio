"""분석 N·O — 비등방성 통제와 상위 주성분 제거.

배경: 마지막 전이의 유효랭크가 ~1.45 로 극히 낮다(이전 단계 실측).
지배적 주성분이 (a) 가짜 신호를 만들었는지 (b) 진짜 신호를 가리고 있었는지 구분한다.

N: 같은 층을 raw / mean-centered / whitened 세 공간에서 평가.
   whitening 은 **TRAIN 에서만** 적합하고 고유값을 정칙화한다.
O: 상위 1·2·5 주성분을 제거한 뒤 centroid/logreg/거리/가지 probe 재평가.
   MAIN 결과를 whitening 으로 대체하는 것이 목적이 아니라 진단이다.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, wcsv, CELLS, ATTACK, PRED_UNSAFE, ALL_ANALYSED, RES, EPS)
from failure_structure.probes import fit_logreg, centroid_dir

MAIN_REPR = ["R1_h", "R2_g"]
REMOVE_K = [0, 1, 2, 5]
RIDGE = 0.05


def eff_rank(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2 / max(len(X) - 1, 1)
    return float(ev.sum() ** 2 / (np.sum(ev ** 2) + EPS))


def evaluate_space(Xtr, ctr, Xte, cte, seed):
    """centroid / logreg / 예측클래스거리 / 가지 probe 를 한 공간에서 평가."""
    ytr, yte = y_error(ctr), y_error(cte)
    w = cell_weights(ctr)
    out = {}
    v = centroid_dir(Xtr, ctr)
    out["centroid"] = auroc(yte, -(Xte @ v))
    m, sc, C, _ = fit_logreg(Xtr, ytr, w, seed)
    out["logreg"] = auroc(yte, m.decision_function(sc.transform(Xte)))
    ma = Xtr[np.isin(ctr, ATTACK)].mean(0); mb = Xtr[~np.isin(ctr, ATTACK)].mean(0)
    pu_te = np.isin(cte, PRED_UNSAFE)
    dpred = np.where(pu_te, np.linalg.norm(Xte - ma, axis=1), np.linalg.norm(Xte - mb, axis=1))
    out["dist_pred_class"] = auroc(yte, dpred)
    for bn, pos, mtr_, mte_ in (("U", "TP", np.isin(ctr, PRED_UNSAFE), pu_te),
                                ("S", "TN", ~np.isin(ctr, PRED_UNSAFE), ~pu_te)):
        ytb = (ctr[mtr_] != pos).astype(int); yeb = (cte[mte_] != pos).astype(int)
        if min(ytb.sum(), (1 - ytb).sum()) < 10 or min(yeb.sum(), (1 - yeb).sum()) < 5:
            out[f"branch_{bn}"] = np.nan; continue
        mb2, sb2, _, _ = fit_logreg(Xtr[mtr_], ytb, cell_weights(ctr[mtr_]), seed)
        out[f"branch_{bn}"] = auroc(yeb, mb2.decision_function(sb2.transform(Xte[mte_])))
    return out


def run():
    D = load(); R = representations(D["h"])
    rows_n, rows_o = [], []
    t0 = time.time()
    for ds in ALL_ANALYSED:
        dm = D["dataset"] == ds
        cd, dd = D["cell"][dm], D["dup"][dm]
        tr, te = group_split(dd, cd, 0)
        ctr, cte = cd[tr], cd[te]
        for rname in MAIN_REPR:
            Xall, lnames = R[rname]
            Xd = Xall[dm]
            for li, ln in enumerate(lnames):
                Xtr = Xd[tr][:, li].astype(np.float64); Xte = Xd[te][:, li].astype(np.float64)
                if np.linalg.norm(Xtr.std(0)) < 1e-8:
                    continue
                er = eff_rank(Xtr)
                mu = Xtr.mean(0)
                # ---- N. raw / centered / whitened ----
                spaces = {"raw": (Xtr, Xte), "centered": (Xtr - mu, Xte - mu)}
                p = PCA(random_state=0).fit(Xtr)
                lam = p.explained_variance_
                keep = lam > lam.max() * 1e-8
                Wm = p.components_[keep].T / np.sqrt(lam[keep] + RIDGE * lam.mean())
                spaces["whitened"] = ((Xtr - mu) @ Wm, (Xte - mu) @ Wm)
                for sname, (A, B) in spaces.items():
                    r = evaluate_space(A, ctr, B, cte, 0)
                    rows_n.append(dict(dataset=ds, repr=rname, layer=ln, space=sname,
                                       effective_rank=er, **r))
                # ---- O. 상위 주성분 제거 ----
                for k in REMOVE_K:
                    if k == 0:
                        A, B = Xtr, Xte
                    else:
                        Vk = p.components_[:k].T
                        A = Xtr - (Xtr - mu) @ Vk @ Vk.T
                        B = Xte - (Xte - mu) @ Vk @ Vk.T
                    r = evaluate_space(A, ctr, B, cte, 0)
                    rows_o.append(dict(dataset=ds, repr=rname, layer=ln, removed_top_pcs=k,
                                       effective_rank=er,
                                       var_removed=float(lam[:k].sum() / lam.sum()) if k else 0.0,
                                       **r))
            print(f"  {ds:20} {rname:10} ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "anisotropy_controls.csv", rows_n)
    wcsv(RES / "top_pc_removal.csv", rows_o)


if __name__ == "__main__":
    run()

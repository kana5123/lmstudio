"""cross-dataset 전이 · LODO · 치환 귀무 (지시문 30~32절).

전이: Dataset A **train** 에서 적합 -> Dataset B **test** 에서 그대로 평가.
      B 의 라벨로 어떤 tuning 도 하지 않는다.  AUROC<0.5 도 뒤집지 않는다.
LODO: usable non-confounded dataset >= 3 일 때만.  pooled 학습 시 데이터셋 균형 가중.
치환 귀무: 데이터셋 내부에서 **GT attack/benign 층 안에서** correctness 배정을 섞는다
      (TP<->FN, FP<->TN 관계를 무작위화하되 GT 구성은 보존).  500회.
"""
import itertools, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, wcsv, CELLS, ATTACK, PRED_UNSAFE, USABLE, CONFOUNDED, MAIN,
    SEEDS, RES, EPS)
from failure_structure.probes import fit_logreg, centroid_dir

MAIN_REPR = ["R1_h", "R2_g"]
NPERM = 500
# 치환 귀무는 비용이 커서 **대표 층만** 돌린다 (선택 기준은 층 이름, 성능 아님)
NULL_LAYERS = {"R1_h": ["h_L6", "h_L9", "h_L11", "h_L12"],
               "R2_g": ["g_L5->L6", "g_L8->L9", "g_L10->L11", "g_L11->L12"]}


def perm_correctness(cell, rng):
    """GT 층화 보존, 정오 배정만 섞기.  attack 안에서 TP<->FN, benign 안에서 TN<->FP."""
    out = np.empty_like(cell)
    atk = np.isin(cell, ATTACK)
    ia, ib = np.flatnonzero(atk), np.flatnonzero(~atk)
    n_tp = int((cell == "TP").sum()); n_tn = int((cell == "TN").sum())
    pa, pb = rng.permutation(ia), rng.permutation(ib)
    out[pa[:n_tp]] = "TP"; out[pa[n_tp:]] = "FN"
    out[pb[:n_tn]] = "TN"; out[pb[n_tn:]] = "FP"
    return out


def build_scorers(Xtr, ctr, seed):
    """A 데이터셋 train 에서 세 종류 점수기를 만든다.  전부 correctness(=incorrect 양성) 방향."""
    ytr = y_error(ctr); w = cell_weights(ctr)
    out = {}
    v = centroid_dir(Xtr, ctr)
    out["centroid"] = lambda X, v=v: -(X @ v)                 # incorrect 가 양성이 되도록 부호
    m, sc, C, _ = fit_logreg(Xtr, ytr, w, seed)
    out["logreg"] = lambda X, m=m, sc=sc: m.decision_function(sc.transform(X))
    p = PCA(32, random_state=0).fit(Xtr)
    sd = np.sqrt(p.explained_variance_) + EPS
    nn = NearestNeighbors(n_neighbors=25).fit(p.transform(Xtr) / sd)
    out["knn"] = lambda X, p=p, sd=sd, nn=nn, ytr=ytr: ytr[nn.kneighbors(p.transform(X) / sd)[1]].mean(1)
    mu = Xtr[np.isin(ctr, ATTACK)].mean(0), Xtr[~np.isin(ctr, ATTACK)].mean(0)
    out["dist_pred_class"] = lambda X, mu=mu: np.linalg.norm(X - mu[0], axis=1)   # placeholder, 아래서 대체
    return out, (m, sc)


def main():
    D = load(); R = representations(D["h"])
    dsets = USABLE + CONFOUNDED
    rows_tx, rows_lodo, rows_null = [], [], []
    t0 = time.time()

    for rname in MAIN_REPR:
        Xall, lnames = R[rname]
        for li, ln in enumerate(lnames):
            # 데이터셋별 train/test 준비
            pack = {}
            for ds in dsets:
                dm = D["dataset"] == ds
                cd, dd = D["cell"][dm], D["dup"][dm]
                tr, te = group_split(dd, cd, 0)
                Xd = Xall[dm][:, li].astype(np.float64)
                if np.linalg.norm(Xd[tr].std(0)) < 1e-8:
                    continue
                pack[ds] = (Xd[tr], cd[tr], Xd[te], cd[te])
            if len(pack) < 2:
                continue

            scorers = {}
            for ds, (Xtr, ctr, _, _) in pack.items():
                sc_, _ = build_scorers(Xtr, ctr, 0)
                mu_a = Xtr[np.isin(ctr, ATTACK)].mean(0)
                mu_b = Xtr[~np.isin(ctr, ATTACK)].mean(0)
                sc_["dist_pred_class"] = (lambda X, C, ma=mu_a, mb=mu_b:
                                          np.where(np.isin(C, PRED_UNSAFE),
                                                   np.linalg.norm(X - ma, axis=1),
                                                   np.linalg.norm(X - mb, axis=1)))
                scorers[ds] = sc_

            # ---------- 전이 ----------
            for a, b in itertools.permutations(pack, 2):
                Xte_b, cte_b = pack[b][2], pack[b][3]
                yb = y_error(cte_b)
                for mname, fn in scorers[a].items():
                    s = fn(Xte_b, cte_b) if mname == "dist_pred_class" else fn(Xte_b)
                    lo, hi = boot_ci(yb, s, seed=0)
                    rows_tx.append(dict(repr=rname, layer=ln, train=a, test=b, method=mname,
                                        auroc_incorrect=auroc(yb, s), auprc_incorrect=auprc(yb, s),
                                        ci_lo=lo, ci_hi=hi, n_test=len(yb),
                                        train_confounded=a in CONFOUNDED,
                                        test_confounded=b in CONFOUNDED))

            # ---------- LODO (비교란 데이터셋만) ----------
            us = [d for d in USABLE if d in pack]
            if len(us) >= 3:
                for held in us:
                    others = [d for d in us if d != held]
                    Xs = np.vstack([pack[o][0] for o in others])
                    Cs = np.concatenate([pack[o][1] for o in others])
                    # 데이터셋 균형 가중 x cell 균형 가중
                    wds = np.concatenate([np.full(len(pack[o][1]), 1.0 / len(pack[o][1]))
                                          for o in others])
                    w = cell_weights(Cs) * wds / wds.mean()
                    ys = y_error(Cs)
                    m, sc, C, _ = fit_logreg(Xs, ys, w, 0)
                    Xte_h, cte_h = pack[held][2], pack[held][3]
                    yh = y_error(cte_h)
                    s = m.decision_function(sc.transform(Xte_h))
                    lo, hi = boot_ci(yh, s, seed=0)
                    rows_lodo.append(dict(repr=rname, layer=ln, held_out=held,
                                          trained_on="+".join(others), method="logreg",
                                          auroc_incorrect=auroc(yh, s), auprc_incorrect=auprc(yh, s),
                                          ci_lo=lo, ci_hi=hi, n_test=len(yh), C=C))

            # ---------- 치환 귀무 (대표 층만) ----------
            if ln in NULL_LAYERS.get(rname, []):
                for a, b in itertools.permutations([d for d in MAIN if d in pack], 2):
                    Xtr_a, ctr_a = pack[a][0], pack[a][1]
                    Xte_b, cte_b = pack[b][2], pack[b][3]
                    yb = y_error(cte_b)
                    rng = np.random.default_rng(0)
                    for mname in ("centroid", "logreg"):
                        obs = auroc(yb, (scorers[a][mname])(Xte_b))
                        nl = []
                        for _ in range(NPERM if mname == "centroid" else 100):
                            cp = perm_correctness(ctr_a, rng)
                            if mname == "centroid":
                                v = centroid_dir(Xtr_a, cp)
                                nl.append(auroc(yb, -(Xte_b @ v)))
                            else:
                                sc0 = StandardScaler().fit(Xtr_a)
                                mm = LogisticRegression(C=0.1, max_iter=200)
                                mm.fit(sc0.transform(Xtr_a), y_error(cp),
                                       sample_weight=cell_weights(cp))
                                nl.append(auroc(yb, mm.decision_function(sc0.transform(Xte_b))))
                        nl = np.array(nl)
                        p = (np.sum(nl >= obs) + 1) / (len(nl) + 1)
                        rows_null.append(dict(repr=rname, layer=ln, train=a, test=b,
                                              method=mname, observed_auroc=obs,
                                              null_mean=float(nl.mean()), null_std=float(nl.std()),
                                              null_p95=float(np.percentile(nl, 95)),
                                              empirical_p=float(p), n_perm=len(nl),
                                              min_attainable_p=1 / (len(nl) + 1)))
            print(f"  {rname:10} {ln:16} ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "cross_dataset_transfer.csv", rows_tx)
    wcsv(RES / "lodo_summary.csv", rows_lodo)
    wcsv(RES / "permutation_null.csv", rows_null)


if __name__ == "__main__":
    main()

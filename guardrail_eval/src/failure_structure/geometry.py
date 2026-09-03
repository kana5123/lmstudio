"""분석 E~I — 거리/반경, Mahalanobis, correct-manifold 복원오차, 국소이웃, 2차항.

전부 "correctness 가 하나의 방향이 아닐 수 있다"는 가정에서 출발한다.
모든 통계(중심, 공분산, PCA, 이웃 pool)는 **TRAIN 에서만** 적합한다.
hyperparameter(k, n_components)는 **TRAIN 내부 검증**에서만 고른다.
TEST 라벨은 어떤 feature 계산에도 쓰지 않는다.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, wcsv, CELLS, ATTACK, PRED_UNSAFE, ALL_ANALYSED, SEEDS,
    INNER_VAL_FRAC, RES, EPS)

MAIN_REPR = ["R1_h", "R2_g"]
K_NN = [5, 10, 25, 50]
K_PCA = [8, 16, 32, 64]
K_QUAD = [32, 64]


def inner(idx_n, seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(idx_n); k = int(idx_n * INNER_VAL_FRAC)
    return p[k:], p[:k]          # inner-train, inner-val


def dists(X, mu, kind):
    if kind == "euclid":
        return np.linalg.norm(X - mu, axis=1)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    mn = mu / (np.linalg.norm(mu) + EPS)
    return 1.0 - Xn @ mn


def run():
    D = load(); R = representations(D["h"])
    rows_dist, rows_mah, rows_rec, rows_knn, rows_quad = [], [], [], [], []
    t0 = time.time()
    for ds in ALL_ANALYSED:
        dm = D["dataset"] == ds
        cell_d, dup_d = D["cell"][dm], D["dup"][dm]
        for rname in MAIN_REPR:
            Xall, lnames = R[rname]
            Xd = Xall[dm]
            for seed in SEEDS:
                tr, te = group_split(dup_d, cell_d, seed)
                ctr, cte = cell_d[tr], cell_d[te]
                ytr, yte = y_error(ctr), y_error(cte)
                ycorr_te = 1 - yte
                atk_tr = np.isin(ctr, ATTACK)
                pu_tr, pu_te = np.isin(ctr, PRED_UNSAFE), np.isin(cte, PRED_UNSAFE)
                for li, ln in enumerate(lnames):
                    Xtr = Xd[tr][:, li].astype(np.float64)
                    Xte = Xd[te][:, li].astype(np.float64)
                    base = dict(dataset=ds, repr=rname, seed=seed, layer=ln,
                                n_test=int(te.sum()))

                    # ---------- E. 거리 / 반경 ----------
                    mu_a, mu_b = Xtr[atk_tr].mean(0), Xtr[~atk_tr].mean(0)
                    for kind in ("euclid", "cosine"):
                        da, db = dists(Xte, mu_a, kind), dists(Xte, mu_b, kind)
                        dpred = np.where(pu_te, da, db)     # 예측한 클래스 중심까지 거리
                        feats = {"dist_attack": da, "dist_benign": db,
                                 "min_dist": np.minimum(da, db),
                                 "abs_diff": np.abs(da - db), "dist_pred_class": dpred}
                        for fn, v in feats.items():
                            lo, hi = boot_ci(yte, v, seed=seed)   # 양성 = incorrect
                            rows_dist.append({**base, "metric": kind, "feature": fn,
                                              "auroc_incorrect": auroc(yte, v),
                                              "auprc_incorrect": auprc(yte, v),
                                              "ci_lo": lo, "ci_hi": hi})

                    # ---------- F. Mahalanobis ----------
                    try:
                        lw_a = LedoitWolf().fit(Xtr[atk_tr]); lw_b = LedoitWolf().fit(Xtr[~atk_tr])
                        ma = lw_a.mahalanobis(Xte); mb = lw_b.mahalanobis(Xte)
                        mp = np.where(pu_te, ma, mb)
                        for fn, v in (("maha_attack", ma), ("maha_benign", mb),
                                      ("maha_min", np.minimum(ma, mb)),
                                      ("maha_pred_class", mp)):
                            lo, hi = boot_ci(yte, v, seed=seed)
                            rows_mah.append({**base, "feature": fn,
                                             "auroc_incorrect": auroc(yte, v),
                                             "auprc_incorrect": auprc(yte, v),
                                             "ci_lo": lo, "ci_hi": hi})
                    except Exception as e:
                        rows_mah.append({**base, "feature": "ERROR", "auroc_incorrect": np.nan,
                                         "auprc_incorrect": np.nan, "ci_lo": np.nan,
                                         "ci_hi": np.nan, "error": str(e)[:80]})

                    # ---------- G. correct-manifold 복원오차 (가지별) ----------
                    for bn, ref, mtr_, mte_ in (("U", "TP", pu_tr, pu_te),
                                                ("S", "TN", ~pu_tr, ~pu_te)):
                        ref_idx = np.flatnonzero(mtr_ & (ctr == ref))
                        if len(ref_idx) < 80 or mte_.sum() < 20:
                            continue
                        it, iv = inner(len(ref_idx), seed)
                        yb_tr = (ctr[mtr_] != ref).astype(int)
                        # k 선택: TRAIN 안에서만 (inner-val 의 branch 표본으로 평가)
                        bt, bv = inner(int(mtr_.sum()), seed + 5)
                        bestk, bestv = K_PCA[0], -np.inf
                        for k in K_PCA:
                            if k >= len(ref_idx):
                                continue
                            p = PCA(k, random_state=0).fit(Xtr[ref_idx[it]])
                            Z = Xtr[mtr_][bv]
                            err = np.linalg.norm(Z - p.inverse_transform(p.transform(Z)), axis=1)
                            a = auroc(yb_tr[bv], err)
                            if not np.isnan(a) and a > bestv:
                                bestv, bestk = a, k
                        p = PCA(bestk, random_state=0).fit(Xtr[ref_idx])
                        Z = Xte[mte_]
                        err = np.linalg.norm(Z - p.inverse_transform(p.transform(Z)), axis=1)
                        yb = (cte[mte_] != ref).astype(int)
                        lo, hi = boot_ci(yb, err, seed=seed)
                        rows_rec.append({**base, "branch": bn, "reference_cell": ref,
                                         "k_pca": bestk, "innerval_auroc": bestv,
                                         "auroc_incorrect": auroc(yb, err),
                                         "auprc_incorrect": auprc(yb, err),
                                         "ci_lo": lo, "ci_hi": hi,
                                         "n_ref_train": len(ref_idx), "n_test_branch": int(mte_.sum())})

                    # ---------- H. 국소 이웃 실패율 ----------
                    # 주의: 전체 거리행렬을 만들면 (n_te, n_tr, 32) 중간배열이 수십 GB 다.
                    # NearestNeighbors 로 k개만 뽑는다 (메모리 상수).
                    pw = PCA(32, random_state=0).fit(Xtr)
                    Ztr = pw.transform(Xtr); Zte = pw.transform(Xte)
                    sd = np.sqrt(pw.explained_variance_) + EPS
                    Ztr_w, Zte_w = Ztr / sd, Zte / sd          # PCA-화이트닝 공간
                    it2, iv2 = inner(len(Ztr_w), seed + 7)
                    nn_in = NearestNeighbors(n_neighbors=max(K_NN)).fit(Ztr_w[it2])
                    _, nb_in = nn_in.kneighbors(Ztr_w[iv2])
                    bestk, bestv = K_NN[0], -np.inf
                    for k in K_NN:
                        a = auroc(ytr[iv2], ytr[it2][nb_in[:, :k]].mean(1))
                        if not np.isnan(a) and a > bestv:
                            bestv, bestk = a, k
                    nn = NearestNeighbors(n_neighbors=bestk).fit(Ztr_w)
                    _, nb = nn.kneighbors(Zte_w)
                    rate = ytr[nb].mean(1)
                    lo, hi = boot_ci(yte, rate, seed=seed)
                    row = {**base, "k": bestk, "innerval_auroc": bestv,
                           "auroc_incorrect": auroc(yte, rate),
                           "auprc_incorrect": auprc(yte, rate),
                           "ci_lo": lo, "ci_hi": hi, "space": "PCA32-whitened"}
                    # 가지별 이웃 실패율 (예측 UNSAFE 는 FP율, 예측 SAFE 는 FN율)
                    for bn, mtr_, mte_ in (("U", pu_tr, pu_te), ("S", ~pu_tr, ~pu_te)):
                        if mtr_.sum() < 60 or mte_.sum() < 20:
                            continue
                        nnb = NearestNeighbors(n_neighbors=min(bestk, int(mtr_.sum()))).fit(Ztr_w[mtr_])
                        _, nbb = nnb.kneighbors(Zte_w[mte_])
                        rb = ytr[mtr_][nbb].mean(1)
                        yb = y_error(cte[mte_])
                        row[f"auroc_branch_{bn}"] = auroc(yb, rb)
                        row[f"n_branch_{bn}_test"] = int(mte_.sum())
                    rows_knn.append(row)

                    # ---------- I. 2차항 (선형 + PC 제곱) ----------
                    for k in K_QUAD:
                        pq = PCA(k, random_state=0).fit(Xtr)
                        Ztr2, Zte2 = pq.transform(Xtr), pq.transform(Xte)
                        Ftr = np.hstack([Ztr2, Ztr2 ** 2]); Fte = np.hstack([Zte2, Zte2 ** 2])
                        scq = StandardScaler().fit(Ftr)
                        w = cell_weights(ctr)
                        mq = LogisticRegression(C=0.1, max_iter=400, solver="lbfgs")
                        mq.fit(scq.transform(Ftr), ytr, sample_weight=w)
                        sq = mq.decision_function(scq.transform(Fte))
                        # 선형만
                        scl = StandardScaler().fit(Ztr2)
                        ml = LogisticRegression(C=0.1, max_iter=400, solver="lbfgs")
                        ml.fit(scl.transform(Ztr2), ytr, sample_weight=w)
                        sl = ml.decision_function(scl.transform(Zte2))
                        lo, hi = boot_ci(yte, sq, seed=seed)
                        rows_quad.append({**base, "n_pc": k,
                                          "auroc_linear_pc": auroc(yte, sl),
                                          "auroc_linear_plus_quad": auroc(yte, sq),
                                          "delta": auroc(yte, sq) - auroc(yte, sl),
                                          "auprc_linear_plus_quad": auprc(yte, sq),
                                          "ci_lo": lo, "ci_hi": hi})
                print(f"  {ds:20} {rname:10} seed{seed} ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "centroid_distance.csv", rows_dist)
    wcsv(RES / "mahalanobis_scores.csv", rows_mah)
    wcsv(RES / "correct_manifold_reconstruction.csv", rows_rec)
    wcsv(RES / "knn_failure_propensity.csv", rows_knn)
    wcsv(RES / "quadratic_probe.csv", rows_quad)


if __name__ == "__main__":
    run()

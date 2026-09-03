"""분석 K·L·M — 실패모드 군집 / 실패 부분공간 / 공분산차 부분공간.

K: INCORRECT(FP+FN)만 모아 군집.  **라벨로 군집을 만들지 않는다** — 군집 후 사후적으로 구성만 본다.
   군집 수는 silhouette + 부트스트랩 안정성으로 평가.  군집에 의미 이름을 붙이지 않는다.
L: 부트스트랩 선형 probe 계수들을 모아 SVD -> 상위 특이벡터가 만드는 부분공간.
   데이터셋 간 비교는 단일 벡터 코사인이 아니라 **주각(principal angles)** 으로.
M: Delta_Sigma = Sigma_incorrect - Sigma_correct 의 절대 고유값 상위 벡터.
   점수 = ||U^T (x - mu)||^2  (평균은 같아도 분산이 다른 경우를 잡음)
"""
import itertools, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, wcsv, CELLS, ATTACK, PRED_UNSAFE, USABLE, CONFOUNDED, SEEDS, RES, OUTA, EPS)

MAIN_REPR = ["R1_h", "R2_g"]
KS = range(2, 11)
NBOOT_PROBE = 100
SUB_K = [1, 2, 4, 8]
COV_K = [1, 2, 4, 8, 16]


def run():
    D = load(); R = representations(D["h"])
    dsets = USABLE + CONFOUNDED
    rows_cl, rows_st, rows_sub, rows_subx, rows_cov = [], [], [], [], []
    subspaces = {}
    t0 = time.time()

    for rname in MAIN_REPR:
        Xall, lnames = R[rname]
        for li, ln in enumerate(lnames):
            for ds in dsets:
                dm = D["dataset"] == ds
                cell_d, dup_d = D["cell"][dm], D["dup"][dm]
                Xd = Xall[dm][:, li].astype(np.float64)
                tr, te = group_split(dup_d, cell_d, 0)
                ctr, cte = cell_d[tr], cell_d[te]
                ytr, yte = y_error(ctr), y_error(cte)
                Xtr, Xte = Xd[tr], Xd[te]
                if np.linalg.norm(Xtr.std(0)) < 1e-8:
                    continue                       # 층 0 처럼 분산 0 인 층은 건너뜀

                # ---------- K. INCORRECT 군집 ----------
                inc = np.flatnonzero(ytr == 1)
                if len(inc) >= 100:
                    p = PCA(32, random_state=0).fit(Xtr[inc])
                    Z = p.transform(Xtr[inc])
                    for k in KS:
                        if k >= len(inc):
                            break
                        km = KMeans(k, n_init=10, random_state=0).fit(Z)
                        sil = float(silhouette_score(Z, km.labels_)) if k > 1 else np.nan
                        # 부트스트랩 안정성: 재표본 군집과 원래 군집의 최대 일치율
                        stab = []
                        for b in range(10):
                            rs = np.random.default_rng(b).choice(len(Z), len(Z), True)
                            km2 = KMeans(k, n_init=5, random_state=b).fit(Z[rs])
                            ct = pd.crosstab(km.labels_[rs], km2.labels_)
                            stab.append(float(ct.max(1).sum() / ct.values.sum()))
                        rows_st.append(dict(dataset=ds, repr=rname, layer=ln, k=k,
                                            silhouette=sil, bootstrap_agreement=float(np.mean(stab)),
                                            n_incorrect=len(inc)))
                        if k == 2:      # 구성은 k=2 에서만 상세 기록
                            for c in range(k):
                                mm = km.labels_ == c
                                sub = ctr[inc][mm]
                                rows_cl.append(dict(dataset=ds, repr=rname, layer=ln, k=k,
                                                    cluster=c, n=int(mm.sum()),
                                                    frac_FP=float((sub == "FP").mean()),
                                                    frac_FN=float((sub == "FN").mean()),
                                                    frac_gt_attack=float(np.isin(sub, ATTACK).mean()),
                                                    frac_pred_unsafe=float(np.isin(sub, PRED_UNSAFE).mean())))

                # ---------- L. 부트스트랩 probe 계수 -> 실패 부분공간 ----------
                sc = StandardScaler().fit(Xtr)
                Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
                W = []
                for b in range(NBOOT_PROBE):
                    rs = np.random.default_rng(b).choice(len(Ztr), len(Ztr), True)
                    if len(set(ytr[rs])) < 2:
                        continue
                    mb = LogisticRegression(C=0.1, max_iter=200, solver="lbfgs")
                    mb.fit(Ztr[rs], ytr[rs], sample_weight=cell_weights(ctr[rs]))
                    W.append(mb.coef_[0])
                if len(W) < 10:
                    continue
                W = np.array(W)
                U, S, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
                for k in SUB_K:
                    B = Vt[:k].T                                   # (d, k)
                    proj_tr, proj_te = Ztr @ B, Zte @ B
                    mm = LogisticRegression(C=1.0, max_iter=300)
                    mm.fit(proj_tr, ytr, sample_weight=cell_weights(ctr))
                    s = mm.decision_function(proj_te)
                    lo, hi = boot_ci(yte, s, seed=0)
                    rows_sub.append(dict(dataset=ds, repr=rname, layer=ln, k=k,
                                         auroc_incorrect=auroc(yte, s),
                                         auprc_incorrect=auprc(yte, s), ci_lo=lo, ci_hi=hi,
                                         sv_ratio_top1=float(S[0] / (S.sum() + EPS)),
                                         n_boot_fits=len(W)))
                subspaces[(rname, ln, ds)] = Vt[:max(SUB_K)].T

                # ---------- M. 공분산차 부분공간 ----------
                Xc, Xi = Xtr[ytr == 0], Xtr[ytr == 1]
                if len(Xc) > 50 and len(Xi) > 50:
                    Sc = np.cov(Xc, rowvar=False); Si = np.cov(Xi, rowvar=False)
                    dS = Si - Sc
                    ev, evec = np.linalg.eigh((dS + dS.T) / 2)
                    order = np.argsort(-np.abs(ev))
                    mu = Xtr.mean(0)
                    for k in COV_K:
                        Uk = evec[:, order[:k]]
                        s = np.sum(((Xte - mu) @ Uk) ** 2, axis=1)
                        lo, hi = boot_ci(yte, s, seed=0)
                        rows_cov.append(dict(dataset=ds, repr=rname, layer=ln, k=k,
                                             auroc_incorrect=auroc(yte, s),
                                             auprc_incorrect=auprc(yte, s),
                                             ci_lo=lo, ci_hi=hi,
                                             top_abs_eig=float(np.abs(ev[order[0]]))))
            print(f"  {rname:10} {ln:16} ({time.time()-t0:.0f}s)", flush=True)

    # ---------- L. 데이터셋 간 부분공간 주각 ----------
    for (rname, ln, a), (rname2, ln2, b) in itertools.combinations(subspaces, 2):
        if rname != rname2 or ln != ln2 or a == b:
            continue
        for k in SUB_K:
            A_, B_ = subspaces[(rname, ln, a)][:, :k], subspaces[(rname, ln, b)][:, :k]
            ang = subspace_angles(A_, B_)
            rows_subx.append(dict(repr=rname, layer=ln, pair=f"{a}|{b}", k=k,
                                  min_angle_deg=float(np.degrees(ang.min())),
                                  mean_angle_deg=float(np.degrees(ang.mean())),
                                  max_angle_deg=float(np.degrees(ang.max())),
                                  mean_cos_principal=float(np.cos(ang).mean())))
    RES.mkdir(parents=True, exist_ok=True); OUTA.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "failure_cluster_summary.csv", rows_cl)
    wcsv(RES / "failure_cluster_stability.csv", rows_st)
    wcsv(RES / "failure_subspace.csv", rows_sub)
    wcsv(RES / "failure_subspace_crossdataset.csv", rows_subx)
    wcsv(RES / "covariance_subspace.csv", rows_cov)
    np.savez_compressed(OUTA / "failure_subspaces.npz",
                        **{f"{r}|{l}|{d}": v for (r, l, d), v in subspaces.items()})


if __name__ == "__main__":
    run()

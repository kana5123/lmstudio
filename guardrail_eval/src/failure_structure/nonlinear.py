"""분석 J — 비선형 probe (과적합 방지를 위해 아주 작은 모델만).

N1: RBF-SVM,  입력 = TRAIN 전용 PCA 32차원
N2: 1층 MLP,  hidden 32, dropout, weight decay, early stopping

모든 hyperparameter 는 **TRAIN 내부 검증**에서만 결정.
같은 표본 수·같은 분할·같은 cell 가중.

용량 통제(33절): 라벨 섞기 학습 + 무작위 가우시안 특징 학습을 **함께** 저장한다.
RBF-SVM 은 O(n^2) 이라 TRAIN 을 최대 SVM_MAX 로 층화 부표본한다(명시).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, wcsv, CELLS, ALL_ANALYSED, SEEDS, INNER_VAL_FRAC, RES, EPS)

MAIN_REPR = ["R1_h", "R2_g"]
N_PC = 32
SVM_MAX = 5000
GAMMAS = ["scale", 0.01]
CS = [1.0, 10.0]
ALPHAS = [1e-3, 1e-2]


def subsample(y, n_max, seed):
    if len(y) <= n_max:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for v in (0, 1):
        i = np.flatnonzero(y == v)
        idx.append(rng.choice(i, min(len(i), n_max // 2), replace=False))
    return np.concatenate(idx)


def fit_eval(Ztr, ytr, wtr, Zva, yva, Zte, yte, seed, kind):
    best, bestm = -np.inf, None
    if kind == "svm":
        s = subsample(ytr, SVM_MAX, seed)
        for C in CS:
            for g in GAMMAS:
                m = SVC(C=C, gamma=g, kernel="rbf")
                m.fit(Ztr[s], ytr[s], sample_weight=wtr[s])
                a = auroc(yva, m.decision_function(Zva))
                if not np.isnan(a) and a > best:
                    best, bestm = a, (C, g)
        m = SVC(C=bestm[0], gamma=bestm[1], kernel="rbf")
        s = subsample(ytr, SVM_MAX, seed)
        m.fit(Ztr[s], ytr[s], sample_weight=wtr[s])
        return auroc(yte, m.decision_function(Zte)), auprc(yte, m.decision_function(Zte)), \
            m.decision_function(Zte), best, str(bestm), int(len(s))
    for al in ALPHAS:
        m = MLPClassifier((32,), alpha=al, max_iter=400, early_stopping=True,
                          n_iter_no_change=10, random_state=seed)
        m.fit(Ztr, ytr)
        a = auroc(yva, m.predict_proba(Zva)[:, 1])
        if not np.isnan(a) and a > best:
            best, bestm = a, al
    m = MLPClassifier((32,), alpha=bestm, max_iter=400, early_stopping=True,
                      n_iter_no_change=10, random_state=seed)
    m.fit(Ztr, ytr)
    p = m.predict_proba(Zte)[:, 1]
    return auroc(yte, p), auprc(yte, p), p, best, str(bestm), int(len(ytr))


def run():
    D = load(); R = representations(D["h"])
    rows = []
    t0 = time.time()
    for ds in ALL_ANALYSED:
        dm = D["dataset"] == ds
        cd, dd = D["cell"][dm], D["dup"][dm]
        for rname in MAIN_REPR:
            Xall, lnames = R[rname]
            Xd = Xall[dm]
            for seed in SEEDS:
                tr, te = group_split(dd, cd, seed)
                ctr, cte = cd[tr], cd[te]
                ytr_all, yte = y_error(ctr), y_error(cte)
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(ytr_all)); k = int(len(ytr_all) * INNER_VAL_FRAC)
                iv, it = perm[:k], perm[k:]
                for li, ln in enumerate(lnames):
                    Xtr = Xd[tr][:, li].astype(np.float64); Xte = Xd[te][:, li].astype(np.float64)
                    if np.linalg.norm(Xtr.std(0)) < 1e-8:
                        continue
                    p = PCA(N_PC, random_state=0).fit(Xtr[it])
                    sc = StandardScaler().fit(p.transform(Xtr[it]))
                    Zt = sc.transform(p.transform(Xtr[it]))
                    Zv = sc.transform(p.transform(Xtr[iv]))
                    Ze = sc.transform(p.transform(Xte))
                    wt = cell_weights(ctr[it])
                    base = dict(dataset=ds, repr=rname, seed=seed, layer=ln,
                                n_train=int(tr.sum()), n_test=int(te.sum()), n_pc=N_PC)
                    for kind in ("svm", "mlp"):
                        a, ap, s_, ival, hp, ntr_used = fit_eval(
                            Zt, ytr_all[it], wt, Zv, ytr_all[iv], Ze, yte, seed, kind)
                        lo, hi = boot_ci(yte, s_, seed=seed)
                        # 용량 통제 1: 라벨 섞기
                        ysh = np.random.default_rng(seed + 99).permutation(ytr_all[it])
                        a_sh = fit_eval(Zt, ysh, wt, Zv, ytr_all[iv], Ze, yte, seed, kind)[0]
                        # 용량 통제 2: 무작위 가우시안 특징 (같은 차원)
                        rg = np.random.default_rng(seed + 7)
                        Rt = rg.standard_normal((len(Zt), N_PC))
                        Rv = rg.standard_normal((len(Zv), N_PC))
                        Re = rg.standard_normal((len(Ze), N_PC))
                        a_rand = fit_eval(Rt, ytr_all[it], wt, Rv, ytr_all[iv], Re, yte,
                                          seed, kind)[0]
                        rows.append({**base, "model": kind, "auroc": a, "auprc": ap,
                                     "ci_lo": lo, "ci_hi": hi, "innerval_auroc": ival,
                                     "hyperparam": hp, "n_train_used": ntr_used,
                                     "auroc_shuffled_label": a_sh,
                                     "auroc_random_features": a_rand})
                print(f"  {ds:20} {rname:10} seed{seed} ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "nonlinear_probe.csv", rows)


if __name__ == "__main__":
    run()

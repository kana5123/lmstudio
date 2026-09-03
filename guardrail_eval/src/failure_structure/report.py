"""§29·§38·§39 — FDR 보정, 요약표 A~E, 최종 판정 근거표."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import norm

from failure_structure.common import RES, MAIN, WEAK, CONFOUNDED, USABLE, fdr_bh, wcsv

def rd(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def auroc_p(auroc_val, n_pos, n_neg):
    """AUROC 가 0.5 초과인지에 대한 근사 p (Hanley-McNeil 분산, 단측).
    부트스트랩 CI 와 별개로 FDR 보정을 위해 층 단위 p 가 필요해서 쓴다."""
    if any(pd.isna([auroc_val, n_pos, n_neg])) or n_pos < 3 or n_neg < 3:
        return np.nan
    a = float(auroc_val)
    q1 = a / (2 - a); q2 = 2 * a ** 2 / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a ** 2) + (n_neg - 1) * (q2 - a ** 2)) / (n_pos * n_neg)
    if var <= 0:
        return np.nan
    return float(1 - norm.cdf((a - 0.5) / np.sqrt(var)))


def main():
    out = {}
    # ---------- 내부(within) 요약 ----------
    cen, log_, lda = rd("centroid_baseline"), rd("logistic_probe"), rd("lda_probe")
    br, qd = rd("branch_specific_probes"), rd("quadratic_probe")
    knn, mah = rd("knn_failure_propensity"), rd("mahalanobis_scores")
    nl = rd("nonlinear_probe"); rec = rd("correct_manifold_reconstruction")
    rows = []
    def add(df, method, col, npos_col=None):
        if df is None or col not in df.columns:
            return
        g = df.groupby(["dataset", "repr", "layer"], dropna=False)
        for (ds, rp, ly), d in g:
            v = d[col].mean()
            npos = d.get("test_n_FP", pd.Series([np.nan])).mean() + d.get("test_n_FN", pd.Series([np.nan])).mean()
            nneg = d.get("test_n_TP", pd.Series([np.nan])).mean() + d.get("test_n_TN", pd.Series([np.nan])).mean()
            rows.append(dict(dataset=ds, repr=rp, layer=ly, method=method,
                             auroc=v, n_pos=npos, n_neg=nneg,
                             ci_lo=d["ci_lo"].mean() if "ci_lo" in d else np.nan,
                             ci_hi=d["ci_hi"].mean() if "ci_hi" in d else np.nan,
                             n_seeds=len(d)))
    add(cen, "centroid", "auroc")
    add(log_, "logreg", "auroc")
    add(lda, "lda", "auroc")
    add(qd, "quadratic", "auroc_linear_plus_quad")
    add(knn, "knn", "auroc_incorrect")
    if mah is not None:
        m2 = mah[mah.feature == "maha_pred_class"]
        add(m2, "mahalanobis_pred_class", "auroc_incorrect")
    if nl is not None:
        for mth in nl["model"].unique():
            add(nl[nl.model == mth], f"nonlinear_{mth}", "auroc")
    if rec is not None:
        add(rec, "manifold_recon", "auroc_incorrect")
    W = pd.DataFrame(rows)
    if not W.empty:
        W["p_raw"] = [auroc_p(a, p, n) for a, p, n in zip(W.auroc, W.n_pos, W.n_neg)]
        for fam, d in W.groupby("method"):
            rej, adj = fdr_bh(d.p_raw.values, 0.05)
            W.loc[d.index, "p_fdr"] = adj; W.loc[d.index, "fdr_sig"] = rej
        W.to_csv(RES / "within_dataset_summary.csv", index=False)
        W.to_csv(RES / "fdr_corrected_results.csv", index=False)
        print(f"저장 within_dataset_summary.csv / fdr_corrected_results.csv ({len(W)}행)")

        # TABLE A
        print("\n=== TABLE A — within-dataset (층별 최댓값이 아니라 층 전체 중앙값) ===")
        piv = W.pivot_table(index=["dataset", "repr"], columns="method", values="auroc",
                            aggfunc="median")
        print(piv.round(3).to_string())

    # ---------- TABLE C 증분 정보 ----------
    inc = rd("incremental_information"); conf = rd("confidence_baselines"); tf = rd("tfidf_control")
    if inc is not None:
        print("\n=== TABLE C — confidence 대비 증분 (층 전체 중 delta 최대 층) ===")
        g = inc.groupby(["dataset", "repr", "layer"]).mean(numeric_only=True).reset_index()
        best = g.loc[g.groupby(["dataset", "repr"]).delta_M2_M0.idxmax()]
        print(best[["dataset", "repr", "layer", "M0_auroc", "M1_hidden_auroc", "M2_auroc",
                    "delta_M2_M0", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string(index=False))
    if conf is not None and tf is not None:
        print("\n=== TABLE (R) — 은닉 vs 어휘 vs confidence ===")
        c = conf[conf.feature == "logit_margin"].groupby("dataset").auroc_incorrect.mean()
        t = tf.groupby("dataset").auroc_incorrect.mean()
        h = W[W.method == "logreg"].groupby("dataset").auroc.max() if not W.empty else None
        cmp = pd.DataFrame({"confidence(logit_margin)": c, "tfidf_text": t,
                            "hidden_logreg(best layer)": h})
        print(cmp.round(3).to_string())

    # ---------- TABLE B/D cross-dataset ----------
    tx = rd("cross_dataset_transfer")
    if tx is not None:
        print("\n=== TABLE B — cross-dataset 전이 (MAIN 쌍만, 층 중앙값) ===")
        s = tx[tx.train.isin(MAIN) & tx.test.isin(MAIN) & (tx.train != tx.test)]
        print(s.pivot_table(index=["train", "test"], columns="method",
                            values="auroc_incorrect", aggfunc="median").round(3).to_string())
    lo = rd("lodo_summary")
    if lo is not None and not lo.empty:
        print("\n=== TABLE — LODO (층 중앙값) ===")
        print(lo.pivot_table(index="held_out", columns="repr",
                             values="auroc_incorrect", aggfunc="median").round(3).to_string())
    nn = rd("permutation_null")
    if nn is not None and not nn.empty:
        print("\n=== 치환 귀무 (cross-dataset 전이) ===")
        print(nn[["repr", "layer", "train", "test", "method", "observed_auroc",
                  "null_mean", "null_p95", "empirical_p"]].round(4).to_string(index=False))
    if br is not None:
        print("\n=== TABLE D — 가지별 (층 중앙값) ===")
        print(br.groupby(["dataset", "repr"])[["auroc_U", "auroc_S"]].median().round(3).to_string())


if __name__ == "__main__":
    main()

"""§37 그림 10종."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from failure_structure.common import RES, PLOT, MAIN, WEAK, CONFOUNDED

def rd(n):
    p = RES / f"{n}.csv"
    return pd.read_csv(p) if p.exists() else None

def lorder(df, col="layer"):
    ls = list(dict.fromkeys(df[col]))
    def key(s):
        import re
        m = re.findall(r"\d+", s)
        return int(m[0]) if m else 0
    return sorted(ls, key=key)

def sty(ds):
    if ds in MAIN: return dict(lw=2.2, ls="-")
    if ds in WEAK: return dict(lw=1.2, ls="--")
    return dict(lw=1.0, ls=":")

def line(ax, df, val, lab):
    for ds, d in df.groupby("dataset"):
        ls = lorder(d)
        g = d.groupby("layer")[val].mean().reindex(ls)
        ax.plot(range(len(ls)), g.values, "o-", label=f"{ds} {lab}".strip(), ms=3, **sty(ds))
    return lorder(df)

def main(repr_="R2_g"):
    PLOT.mkdir(parents=True, exist_ok=True)
    def sub(df):
        return df[df["repr"] == repr_] if df is not None and "repr" in df.columns else df

    # 1. layerwise probe comparison
    cen, log_, lda, qd = map(lambda n: sub(rd(n)),
                             ["centroid_baseline", "logistic_probe", "lda_probe", "quadratic_probe"])
    if cen is not None and log_ is not None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ls = None
        for df, val, lab, mk in ((cen, "auroc", "centroid", "s--"),
                                 (log_, "auroc", "logreg", "o-"),
                                 (lda, "auroc", "LDA", "^:"),
                                 (qd, "auroc_linear_plus_quad", "quad", "v-.")):
            if df is None or val not in df.columns: continue
            for ds, d in df.groupby("dataset"):
                ls = lorder(d)
                g = d.groupby("layer")[val].mean().reindex(ls)
                ax.plot(range(len(ls)), g.values, mk, ms=3, alpha=.85,
                        label=f"{ds}·{lab}", **sty(ds))
        ax.axhline(.5, color="gray", ls=":")
        if ls: ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls, rotation=90, fontsize=6)
        ax.set_ylabel("held-out AUROC (incorrect = positive)")
        ax.set_title(f"1. Layer-wise probe comparison [{repr_}]")
        ax.legend(fontsize=5, ncol=4); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"layerwise_probe_comparison_{repr_}.png", dpi=145); plt.close()

    # 2. linear vs nonlinear
    nl = sub(rd("nonlinear_probe"))
    if nl is not None and log_ is not None:
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ls = line(ax, log_, "auroc", "logreg")
        for mdl, d in nl.groupby("model"):
            for ds, dd in d.groupby("dataset"):
                l2 = lorder(dd)
                g = dd.groupby("layer")["auroc"].mean().reindex(l2)
                ax.plot(range(len(l2)), g.values, "^--", ms=3, alpha=.8,
                        label=f"{ds}·{mdl}", **sty(ds))
        ax.axhline(.5, color="gray", ls=":")
        ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls, rotation=90, fontsize=6)
        ax.set_ylabel("AUROC"); ax.set_title(f"2. Linear vs nonlinear [{repr_}]")
        ax.legend(fontsize=5, ncol=3); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"linear_vs_nonlinear_{repr_}.png", dpi=145); plt.close()

    # 3. branch specific
    br = sub(rd("branch_specific_probes"))
    if br is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
        for ax, c, t in zip(axes, ["auroc_U", "auroc_S"],
                            ["Probe-U: TP vs FP (pred UNSAFE)", "Probe-S: TN vs FN (pred SAFE)"]):
            ls = line(ax, br, c, "")
            ax.axhline(.5, color="gray", ls=":")
            ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls, rotation=90, fontsize=6)
            ax.set_title(t, fontsize=10); ax.grid(alpha=.3)
        axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=6)
        plt.tight_layout(); plt.savefig(PLOT / f"branch_specific_{repr_}.png", dpi=145); plt.close()

    # 4. distance / density
    kn, mh, rc = map(lambda n: sub(rd(n)),
                     ["knn_failure_propensity", "mahalanobis_scores", "correct_manifold_reconstruction"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ls = None
    if kn is not None: ls = line(ax, kn, "auroc_incorrect", "kNN")
    if mh is not None:
        m2 = mh[mh.feature == "maha_pred_class"]
        if not m2.empty: ls = line(ax, m2, "auroc_incorrect", "Maha")
    if rc is not None and not rc.empty:
        ls = line(ax, rc.groupby(["dataset", "layer"], as_index=False).auroc_incorrect.mean(),
                  "auroc_incorrect", "recon")
    ax.axhline(.5, color="gray", ls=":")
    if ls: ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls, rotation=90, fontsize=6)
    ax.set_ylabel("AUROC"); ax.set_title(f"4. Distance / density structure [{repr_}]")
    ax.legend(fontsize=5, ncol=3); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / f"distance_density_results_{repr_}.png", dpi=145); plt.close()

    # 5. top-PC removal
    tp = sub(rd("top_pc_removal"))
    if tp is not None:
        fig, ax = plt.subplots(figsize=(11, 4.4))
        for ds, d in tp.groupby("dataset"):
            for k, dd in d.groupby("removed_top_pcs"):
                l2 = lorder(dd)
                g = dd.groupby("layer")["logreg"].mean().reindex(l2)
                ax.plot(range(len(l2)), g.values, "o-", ms=3, alpha=.8,
                        label=f"{ds} rm{k}", **sty(ds))
        ax.axhline(.5, color="gray", ls=":")
        ax.set_ylabel("logreg AUROC"); ax.set_title(f"5. Top-PC removal [{repr_}]")
        ax.legend(fontsize=5, ncol=4); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"top_pc_removal_{repr_}.png", dpi=145); plt.close()

    # 6. hidden vs confidence
    inc = sub(rd("incremental_information"))
    if inc is not None:
        fig, ax = plt.subplots(figsize=(11, 4.4))
        for ds, d in inc.groupby("dataset"):
            l2 = lorder(d)
            for c, mk in (("M0_auroc", "s--"), ("M1_hidden_auroc", "o-"), ("M2_auroc", "^-.")):
                g = d.groupby("layer")[c].mean().reindex(l2)
                ax.plot(range(len(l2)), g.values, mk, ms=3, alpha=.85,
                        label=f"{ds}·{c}", **sty(ds))
        ax.axhline(.5, color="gray", ls=":")
        ax.set_ylabel("AUROC"); ax.set_title(f"6. Hidden vs confidence (M0/M1/M2) [{repr_}]")
        ax.legend(fontsize=5, ncol=3); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"hidden_vs_confidence_{repr_}.png", dpi=145); plt.close()

    # 7/8. cross-dataset + LODO heatmaps
    tx = sub(rd("cross_dataset_transfer"))
    if tx is not None:
        s = tx[tx.method.isin(["centroid", "logreg", "knn", "dist_pred_class"])]
        pv = s.pivot_table(index=["train", "test"], columns="method",
                           values="auroc_incorrect", aggfunc="median")
        fig, ax = plt.subplots(figsize=(6, 0.42 * len(pv) + 2))
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=45, fontsize=7)
        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels([f"{a}→{b}" for a, b in pv.index], fontsize=6)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=.8); ax.set_title(f"7. Cross-dataset transfer [{repr_}]", fontsize=9)
        plt.tight_layout(); plt.savefig(PLOT / f"crossdataset_transfer_heatmap_{repr_}.png", dpi=145); plt.close()
    lo = sub(rd("lodo_summary"))
    if lo is not None and not lo.empty:
        pv = lo.pivot_table(index="held_out", columns="layer", values="auroc_incorrect")
        pv = pv[lorder(lo)]
        fig, ax = plt.subplots(figsize=(0.55 * pv.shape[1] + 3, 0.6 * pv.shape[0] + 2))
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.8); ax.set_title(f"8. LODO [{repr_}]", fontsize=9)
        plt.tight_layout(); plt.savefig(PLOT / f"lodo_heatmap_{repr_}.png", dpi=145); plt.close()

    # 9. failure cluster stability
    st = sub(rd("failure_cluster_stability"))
    if st is not None and not st.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, c, t in zip(axes, ["silhouette", "bootstrap_agreement"],
                            ["silhouette", "bootstrap agreement"]):
            for ds, d in st.groupby("dataset"):
                g = d.groupby("k")[c].mean()
                ax.plot(g.index, g.values, "o-", label=ds, **sty(ds))
            ax.set_xlabel("k (clusters of INCORRECT only)"); ax.set_title(t, fontsize=10)
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        plt.suptitle(f"9. Failure-mode clustering [{repr_}]", y=1.02)
        plt.tight_layout(); plt.savefig(PLOT / f"failure_cluster_pca_{repr_}.png", dpi=145,
                                        bbox_inches="tight"); plt.close()

    # 10. subspace similarity
    sx = sub(rd("failure_subspace_crossdataset"))
    if sx is not None and not sx.empty:
        fig, ax = plt.subplots(figsize=(11, 4.4))
        for (pair, k), d in sx.groupby(["pair", "k"]):
            if k != 4: continue
            l2 = lorder(d)
            g = d.groupby("layer")["mean_cos_principal"].mean().reindex(l2)
            ax.plot(range(len(l2)), g.values, "o-", ms=3, label=f"{pair} k=4")
        ax.set_ylabel("mean cos of principal angles"); ax.set_ylim(0, 1)
        ax.set_title(f"10. Failure-subspace similarity across datasets [{repr_}]")
        ax.legend(fontsize=6); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"failure_subspace_similarity_{repr_}.png", dpi=145); plt.close()
    print(f"저장 -> {PLOT}  ({repr_})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "R2_g")

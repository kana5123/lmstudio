"""§20 그림 8종."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, PLOT = ROOT / "results/direction_correctness", ROOT / "plots/direction_correctness"
MAIN = ["wildjailbreak", "promptshield_test"]
OTHER = ["questionset", "jailbreaksovertime"]


def lay(df):
    t = list(dict.fromkeys(df["layer"]))
    return [x for x in t if not x.endswith("L0->L1") and x != "L0"]


def sty(ds):
    return dict(lw=2.2, ls="-") if ds in MAIN else dict(lw=1.1, ls=":")


def main(tag="move"):
    PLOT.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(RES / f"correctness_auroc_by_layer__{tag}.csv")
    S = pd.read_csv(RES / f"subgroup_auroc_by_layer__{tag}.csv")
    E = pd.read_csv(RES / f"effect_directions_by_layer__{tag}.csv")
    L = pd.read_csv(RES / f"label_leakage_by_layer__{tag}.csv")
    tr = lay(A)
    idx = {t: i for i, t in enumerate(tr)}

    def prep(df, val):
        g = df.groupby(["dataset", "layer"])[val].agg(["mean", "std"]).reset_index()
        return g[g.layer.isin(tr)]

    # 1. correctness AUROC
    fig, ax = plt.subplots(figsize=(10, 4.6))
    g = prep(A, "auroc_correctness")
    for ds, d in g.groupby("dataset"):
        d = d.set_index("layer").loc[tr]
        ax.plot([idx[t] for t in tr], d["mean"], "o-", label=ds, **sty(ds))
        ax.fill_between([idx[t] for t in tr], d["mean"] - d["std"], d["mean"] + d["std"], alpha=.15)
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
    ax.set_ylabel("held-out correctness AUROC (delta_CORR)"); ax.set_xlabel("layer")
    ax.set_title(f"1. Correctness AUROC on held-out  [{tag}]  (mean±sd over 5 seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / f"1_correctness_auroc__{tag}.png", dpi=145); plt.close()

    # 2. subgroup
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, col, ttl in zip(axes, ["auroc_danger_TPvFP", "auroc_safe_TNvFN"],
                            ["predicted-danger subset: TP vs FP", "predicted-safe subset: TN vs FN"]):
        g = prep(S, col)
        for ds, d in g.groupby("dataset"):
            d = d.set_index("layer").loc[tr]
            ax.plot([idx[t] for t in tr], d["mean"], "o-", label=ds, **sty(ds))
        ax.axhline(.5, color="gray", ls=":")
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
    axes[0].set_ylabel("AUROC using the SAME delta_CORR"); axes[0].legend(fontsize=8)
    plt.suptitle(f"2. One direction must work in BOTH subsets  [{tag}]", y=1.02)
    plt.tight_layout(); plt.savefig(PLOT / f"2_subgroup_auroc__{tag}.png", dpi=145,
                                    bbox_inches="tight"); plt.close()

    # 3. ||delta_CORR||
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for ds, d in prep(E, "norm_CORR").groupby("dataset"):
        d = d.set_index("layer").loc[tr]
        ax.plot([idx[t] for t in tr], d["mean"], "o-", label=f"{ds}  ||CORR||", **sty(ds))
    for ds, d in prep(E, "norm_LABEL").groupby("dataset"):
        if ds not in MAIN:
            continue
        d = d.set_index("layer").loc[tr]
        ax.plot([idx[t] for t in tr], d["mean"], "s--", alpha=.5, label=f"{ds}  ||LABEL||")
    ax.set_yscale("log"); ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
    ax.set_ylabel("‖direction‖ (log)"); ax.set_title(f"3. Magnitude of delta_CORR vs delta_LABEL  [{tag}]")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / f"3_direction_norms__{tag}.png", dpi=145); plt.close()

    # 4. split reproducibility
    f = RES / f"split_reproducibility_by_layer__{tag}.csv"
    if f.exists():
        R = pd.read_csv(f); R = R[(R.direction == "CORR") & (R.layer.isin(tr))]
        fig, ax = plt.subplots(figsize=(10, 4.4))
        for ds, d in R.groupby("dataset"):
            d = d.set_index("layer").loc[tr]
            ax.plot([idx[t] for t in tr], d["mean"], "o-", label=ds, **sty(ds))
            ax.fill_between([idx[t] for t in tr], d["min"], d["max"], alpha=.12)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
        ax.set_ylabel("cos between seeds"); ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"4. Split reproducibility of delta_CORR  [{tag}]")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"4_split_reproducibility__{tag}.png", dpi=145); plt.close()

    # 5. cross-dataset cos
    f = RES / f"cross_dataset_reproducibility_by_layer__{tag}.csv"
    if f.exists():
        C = pd.read_csv(f); C = C[(C.direction == "CORR") & (C.layer.isin(tr))]
        fig, ax = plt.subplots(figsize=(10, 4.4))
        for pr, d in C.groupby("pair"):
            d = d.set_index("layer").loc[tr]
            hl = all(x in MAIN for x in pr.split("|"))
            ax.plot([idx[t] for t in tr], d["cos"], "o-" if hl else "s:",
                    lw=2.4 if hl else 1.0, label=pr)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
        ax.set_ylabel("cos(delta_CORR_A, delta_CORR_B)"); ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"5. Cross-dataset direction agreement  [{tag}]")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"5_crossdataset_cosine__{tag}.png", dpi=145); plt.close()

    # 6/7. correctness vs label scatter
    for k, nm in (("CORR", "6_deltaCORR_correctness_vs_label"),
                  ("LABEL", "7_deltaLABEL_label_vs_correctness")):
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        g = L.groupby(["dataset", "layer"])[[f"{k}_auroc_correctness", f"{k}_auroc_label"]].mean().reset_index()
        g = g[g.layer.isin(tr)]
        for ds, d in g.groupby("dataset"):
            ax.scatter(d[f"{k}_auroc_label"], d[f"{k}_auroc_correctness"],
                       s=60 if ds in MAIN else 25, alpha=.85, label=ds)
        ax.axhline(.5, color="gray", ls=":"); ax.axvline(.5, color="gray", ls=":")
        ax.set_xlabel(f"AUROC: {k} direction -> attack vs benign")
        ax.set_ylabel(f"AUROC: {k} direction -> correct vs incorrect")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{'6' if k=='CORR' else '7'}. delta_{k} disentanglement  [{tag}]\n"
                     f"(ideal for CORR: top-centre; for LABEL: right-centre)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"{nm}__{tag}.png", dpi=145); plt.close()

    # 8. permutation null
    f = RES / f"permutation_null_by_layer__{tag}.csv"
    if f.exists():
        P = pd.read_csv(f); P = P[P.layer.isin(tr)]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        g = P.groupby(["dataset", "layer"])[["observed_auroc", "null_mean", "null_p95"]].mean().reset_index()
        for ds, d in g.groupby("dataset"):
            d = d.set_index("layer").loc[tr]
            ax.plot([idx[t] for t in tr], d["observed_auroc"], "o-", label=f"{ds} observed", **sty(ds))
            ax.plot([idx[t] for t in tr], d["null_p95"], "--", alpha=.45, label=f"{ds} null p95")
        ax.axhline(.5, color="gray", ls=":")
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
        ax.set_ylabel("correctness AUROC"); ax.set_title(
            f"8. Observed vs permutation null  [{tag}]  (500 perms, min p = 2.0e-3)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(PLOT / f"8_permutation_null__{tag}.png", dpi=145); plt.close()
    print(f"저장 -> {PLOT}  ({tag})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "move")

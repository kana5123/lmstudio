"""§22 그림."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RD, RX = ROOT / "results/direction_debug", ROOT / "results/direction_crossdataset"
PLOT = ROOT / "plots/direction_debug"
MAIN = ["wildjailbreak", "promptshield_test", "questionset"]


def order(df):
    t = [c for c in df["transition"].unique() if c != "L0->L1"]
    return sorted(t, key=lambda s: int(s.split("->")[0][1:]))


def main():
    PLOT.mkdir(parents=True, exist_ok=True)

    # 1. branch U vs S
    b = pd.read_csv(RD / "branch_alignment.csv")
    tr = order(b)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for ds, g in b.groupby("dataset"):
        g = g.set_index("transition").loc[tr]
        # 부트스트랩 구간이 점추정을 감쌀 보장이 없어(치우친 통계량) errorbar 대신 밴드로 그린다
        ax.plot(range(len(tr)), g["cos_U_S"], "o-", label=ds,
                lw=2 if ds in MAIN else 1, ls="-" if ds in MAIN else ":")
        ax.fill_between(range(len(tr)), g["boot_ci_lo"], g["boot_ci_hi"], alpha=.15)
        if ds == MAIN[0]:
            ax.fill_between(range(len(tr)), -g["null_p95"], g["null_p95"], color="k",
                            alpha=.12, label="null 95% (shuffled correctness)")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
    ax.set_ylabel("cos(delta_U, delta_S)"); ax.set_xlabel("layer transition")
    ax.set_title("Do the UNSAFE and SAFE branches move the same way for correct-vs-error?")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / "branch_U_vs_S_cosine.png", dpi=145); plt.close()

    # 2. effect cosines
    e = pd.read_csv(RD / "effect_alignment_by_layer.csv")
    tr = order(e)
    dss = [d for d in MAIN if d in set(e["dataset"])]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = e[e.dataset == ds].set_index("transition").loc[tr]
        for c, lab in (("cos_U_GT", "cos(U, GT)"), ("cos_U_PRED", "cos(U, PRED)"),
                       ("cos_U_CORR", "cos(U, CORR)"), ("cos_U_S", "cos(U, S)")):
            ax.plot(range(len(tr)), g[c], "o-", label=lab, ms=4)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=90, fontsize=7)
        ax.set_title(ds, fontsize=10); ax.grid(alpha=.3); ax.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("cosine"); axes[0][0].legend(fontsize=7)
    plt.suptitle("What is the TP-FP direction actually aligned with?", y=1.02)
    plt.tight_layout(); plt.savefig(PLOT / "effect_cosines_by_layer.png", dpi=145,
                                    bbox_inches="tight"); plt.close()

    # 3. anisotropy
    a = pd.read_csv(RD / "anisotropy_by_layer.csv")
    tr = order(a)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for ds, g in a.groupby("dataset"):
        g = g.set_index("transition").loc[tr]
        ax.plot(range(len(tr)), g["effective_rank"], "o-", label=ds)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=45)
    ax.set_ylabel("effective rank  (tr Σ)²/tr(Σ²),  dim = 768")
    ax.set_title("Movement vectors live in a very low-dimensional subspace")
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / "anisotropy_effective_rank.png", dpi=145); plt.close()

    # 4. confidence correlation
    c = pd.read_csv(RD / "confidence_correlations.csv")
    tr = order(c)
    dss = [d for d in MAIN if d in set(c["dataset"])]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for k in ("U", "S", "GT", "PRED", "CORR"):
            g = c[(c.dataset == ds) & (c.direction == k)].set_index("transition").loc[tr]
            ax.plot(range(len(tr)), g["spearman_logit_margin"], "o-", label=f"v_{k}", ms=4)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=90, fontsize=7)
        ax.set_title(ds, fontsize=10); ax.grid(alpha=.3); ax.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("Spearman(projection, PG2 logit margin)")
    axes[0][0].legend(fontsize=7)
    plt.suptitle("Control: how much is each direction just PG2 confidence?", y=1.02)
    plt.tight_layout(); plt.savefig(PLOT / "confidence_correlation_by_layer.png", dpi=145,
                                    bbox_inches="tight"); plt.close()

    # 5. cross-dataset CORR heatmap
    if (RX / "pairwise_direction_cosines.csv").exists():
        p = pd.read_csv(RX / "pairwise_direction_cosines.csv")
        for eff in ("CORR", "U", "S"):
            s = p[p.effect == eff]
            if s.empty:
                continue
            tr2 = order(s)
            pv = s.pivot_table(index="pair", columns="transition", values="cos")[tr2]
            fig, ax = plt.subplots(figsize=(1.0 * len(tr2) + 3, 0.5 * len(pv) + 2.5))
            im = ax.imshow(pv.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(tr2))); ax.set_xticklabels(tr2, rotation=90, fontsize=7)
            ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i,j]:+.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, shrink=.7)
            ax.set_title(f"cross-dataset cos(delta_{eff})")
            plt.tight_layout()
            nm = ("crossdataset_corr_direction_heatmap.png" if eff == "CORR"
                  else f"crossdataset_{eff}_direction_heatmap.png")
            plt.savefig(PLOT / nm, dpi=145); plt.close()

    # 6. transfer heatmap
    if (RX / "cross_dataset_transfer.csv").exists():
        t = pd.read_csv(RX / "cross_dataset_transfer.csv")
        t = t[t.fit_dataset.isin(MAIN) & t.test_dataset.isin(MAIN)]
        if not t.empty:
            tr2 = order(t)
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), squeeze=False)
            for ax, k in zip(axes[0], ("U", "S", "CORR")):
                s = t[t.direction == k]
                if s.empty:
                    continue
                pv = s.pivot_table(index=["fit_dataset", "test_dataset"],
                                   columns="transition", values="auroc")
                pv = pv[[c for c in tr2 if c in pv.columns]]
                im = ax.imshow(pv.values, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
                ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=90, fontsize=6)
                ax.set_yticks(range(pv.shape[0]))
                ax.set_yticklabels([f"{a}→{b}" for a, b in pv.index], fontsize=6)
                ax.set_title(f"delta_{k} transfer AUROC")
                fig.colorbar(im, ax=ax, shrink=.7)
            plt.tight_layout(); plt.savefig(PLOT / "crossdataset_transfer_heatmap.png", dpi=145)
            plt.close()
    print(f"저장 -> {PLOT}")


if __name__ == "__main__":
    main()

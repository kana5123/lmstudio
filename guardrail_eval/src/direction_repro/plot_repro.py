"""재현성 그림."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, PLOT = ROOT / "results/direction_repro", ROOT / "plots/direction_repro"


def main():
    PLOT.mkdir(parents=True, exist_ok=True)
    xd = pd.read_csv(RES / "cross_dataset_cosine.csv")
    nu = pd.read_csv(RES / "null_cosine_distribution.csv")
    ip = pd.read_csv(RES / "internal_partition_cosine.csv")
    cols = [c for c in ip.columns if c.startswith("cos_")]
    tr = [c[len("cos_"):] for c in cols]
    ix = list(range(1, len(tr)))                       # 임베딩->L1 제외

    fig, ax = plt.subplots(figsize=(10, 4.8))
    M = ip[cols].values[:, ix]
    ax.fill_between(range(len(ix)), M.min(0), M.max(0), alpha=.25, color="#2471a3",
                    label="internal partitions (min–max)")
    ax.plot(range(len(ix)), np.median(M, 0), "o-", color="#2471a3", label="internal median")
    ax.plot(range(len(ix)), xd["cos"].values[ix], "s-", color="#c0392b", lw=2.2,
            label="cross-dataset  WildJailbreak vs JailbreaksOverTime")
    ax.plot(range(len(ix)), nu["null_abs_cos_mean"].values[ix], "k--",
            label="empirical null (shuffled labels)")
    ax.plot(range(len(ix)), -nu["null_abs_cos_mean"].values[ix], "k--", alpha=.4)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(range(len(ix))); ax.set_xticklabels([tr[i] for i in ix], rotation=45)
    ax.set_ylabel("cos(direction_a, direction_b)"); ax.set_xlabel("layer transition")
    ax.set_title("Does the TP-vs-FP direction reproduce?  (directions fit on train only)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylim(-0.75, 1.05)
    plt.tight_layout(); plt.savefig(PLOT / "direction_reproducibility.png", dpi=145); plt.close()

    t2 = pd.read_csv(RES / "cross_dataset_transfer_auroc.csv")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(range(len(ix)), t2["auroc_own_direction"].values[ix], "o-", label="own (WildJailbreak) direction")
    ax.plot(range(len(ix)), t2["auroc_jot_direction"].values[ix], "s--", color="#c0392b",
            label="JailbreaksOverTime direction transferred")
    ax.axhline(.5, color="gray", ls=":", label="chance")
    ax.set_xticks(range(len(ix))); ax.set_xticklabels([tr[i] for i in ix], rotation=45)
    ax.set_ylabel("AUROC on WildJailbreak held-out"); ax.set_xlabel("layer transition")
    ax.set_title("Cross-dataset transfer of the direction"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / "cross_dataset_transfer.png", dpi=145); plt.close()
    print(f"저장 -> {PLOT}")


if __name__ == "__main__":
    main()

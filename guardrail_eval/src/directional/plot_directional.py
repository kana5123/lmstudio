"""방향성 정렬 그림 (지시문 21절).  판단 근거는 CSV 이고 그림은 전달용이다."""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/directional_alignment"
ART = ROOT / "artifacts/directional_alignment"
PLOT = ROOT / "plots/directional_alignment"


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    PLOT.mkdir(parents=True, exist_ok=True)
    (PLOT / "tp_fp_projection_distributions").mkdir(exist_ok=True)
    rows = [r for r in read(RES / "global_alignment_all_splits.csv")
            if r["main_analysis"] == "True" and r["subset"] == "all"]
    splits = ["ver_train", "ver_dev", "eval_val", "eval_test"]
    trans = [r["transition"] for r in rows if r["split"] == "ver_train"]
    x = np.arange(len(trans))

    # --- 층별 AUROC ---
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s in splits:
        y = [float(r["q_auroc"]) for r in rows if r["split"] == s]
        lo = [float(r["q_auroc_ci_lo"]) for r in rows if r["split"] == s]
        hi = [float(r["q_auroc_ci_hi"]) for r in rows if r["split"] == s]
        style = "o-" if s != "ver_train" else "s--"
        ax.plot(x, y, style, label=s + (" (fit split)" if s == "ver_train" else ""))
        if s != "ver_train":
            ax.fill_between(x, lo, hi, alpha=.15)
    # 출처 판별 AUROC 를 같이 그려 교란을 눈으로 보이게
    try:
        cmp = [r for r in read(RES / "tpfp_vs_source_auroc.csv") if r["split"] == "eval_test"]
        ax.plot(x, [float(r["auroc_source"]) for r in cmp], "k:", lw=2,
                label="eval_test: same q predicting SOURCE (wildchat vs not)")
    except FileNotFoundError:
        pass
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xticks(x); ax.set_xticklabels(trans, rotation=45, ha="right")
    ax.set_ylabel("AUROC of q (TP vs FP)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Layer-wise directional alignment: q = <v_U, g> - tau")
    plt.tight_layout(); plt.savefig(PLOT / "layerwise_auroc.png", dpi=140); plt.close()

    # --- 층별 효과크기 ---
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s in splits:
        ax.plot(x, [float(r["q_cohens_d"]) for r in rows if r["split"] == s],
                "o-" if s != "ver_train" else "s--", label=s)
    ax.set_xticks(x); ax.set_xticklabels(trans, rotation=45, ha="right")
    ax.set_ylabel("Cohen's d (TP vs FP on q)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Layer-wise effect size")
    plt.tight_layout(); plt.savefig(PLOT / "layerwise_effect_size.png", dpi=140); plt.close()

    # --- q 분포 ---
    for s in ("eval_val", "eval_test"):
        z = np.load(ART / f"proj_{s}.npz", allow_pickle=True)
        q, y = z["q"], z["y"]
        for li in range(1, q.shape[1]):
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            ax.hist(q[y == 1, li], bins=50, alpha=.6, label="TP", color="#c0392b", density=True)
            ax.hist(q[y == 0, li], bins=50, alpha=.6, label="FP", color="#2471a3", density=True)
            ax.axvline(0, color="k", ls=":", lw=1)
            ax.set_title(f"{s}  L{li}->L{li+1}   q distribution"); ax.legend(fontsize=8)
            ax.set_xlabel("q = <v_U, g> - tau")
            plt.tight_layout()
            plt.savefig(PLOT / "tp_fp_projection_distributions" / f"{s}_L{li}_L{li+1}.png", dpi=110)
            plt.close()
    print(f"저장 -> {PLOT}")


if __name__ == "__main__":
    main()

"""최종 결과 그림.  판단 근거는 표(json)이고 그림은 전달용이다."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES, PLOT = ROOT / "artifacts/results", ROOT / "artifacts/plots"


def main():
    d = json.load(open(RES / "final_guard.json"))
    ks = ["1pct", "0.5pct", "0.1pct"]
    lbl = ["1%", "0.5%", "0.1%"]
    names = [k[len("cascade_"):] for k in d
             if k.startswith("cascade_") and isinstance(d[k], dict) and "mean_std" in d[k]]
    PLOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, mode, title in ((axes[0], "cascade", "pure replacement (verifier replaces PG2 score)"),
                            (axes[1], "blend", "blend (w and threshold both chosen on eval_val)")):
        base = [d["PG2_raw"][f"recall@{k}"] for k in ks]
        ci = np.array([d["PG2_raw"][f"recall_ci@{k}"] for k in ks]).T
        x = np.arange(len(ks))
        ax.errorbar(x, base, yerr=[np.array(base) - ci[0], ci[1] - np.array(base)],
                    fmt="ko-", lw=2.5, capsize=4, label="PromptGuard2 alone", zorder=5)
        for i, n in enumerate(sorted(names)):
            key = f"{mode}_{n}"
            if key not in d:
                continue
            y = [d[key]["mean_std"][f"recall@{k}"][0] for k in ks]
            e = [d[key]["mean_std"][f"recall@{k}"][1] for k in ks]
            ax.errorbar(x + (i + 1) * .04 - .1, y, yerr=e, fmt="o--", alpha=.8, capsize=3, label=n)
        ax.axhline(d["cascade_recall_ceiling"], color="gray", ls=":",
                   label=f"cascade ceiling {d['cascade_recall_ceiling']:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_xlabel("target FPR")
        ax.set_ylabel("Recall (eval_test)"); ax.set_title(title); ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(PLOT / "recall_at_fpr.png", dpi=140); plt.close()
    print(f"저장 -> {PLOT/'recall_at_fpr.png'}")

    lp = RES / "latency_cuda_bs1.json"
    if not lp.exists():
        print("지연시간 결과가 아직 없어 두 번째 그림은 건너뜀")
        return
    lat = json.load(open(lp))

    def rec(name):
        return d.get(f"cascade_{name}", {}).get("mean_std", {}).get("recall@1pct", [float("nan")])[0]

    pts = [("PG2 alone", lat["N512"]["pg2_forward"]["mean_ms"], d["PG2_raw"]["recall@1pct"]),
           ("+hidden light verifier", lat["N512"]["path_A_total"]["mean_ms"], rec("B2_delta_only")),
           ("+DecompX verifier", lat["N512"]["path_B_total"]["mean_ms"], rec("B7_full"))]
    if "path_mdeberta_total" in lat["N512"]:
        pts.append(("+mDeBERTa verifier", lat["N512"]["path_mdeberta_total"]["mean_ms"],
                    rec("B8_mdeberta")))
    if not np.isnan(rec("B9_distilled")):
        pts.append(("+distilled verifier", lat["N512"]["path_A_total"]["mean_ms"], rec("B9_distilled")))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for n, x, y in pts:
        if not np.isnan(y):
            ax.scatter(x, y, s=90)
            ax.annotate(n, (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("per-sample latency (ms, N=512, batch 1, GPU)")
    ax.set_ylabel("Recall @ 1% FPR"); ax.grid(alpha=.3)
    ax.set_title("accuracy vs latency")
    plt.tight_layout(); plt.savefig(PLOT / "latency_vs_recall.png", dpi=140); plt.close()
    print(f"저장 -> {PLOT/'latency_vs_recall.png'}")


if __name__ == "__main__":
    main()

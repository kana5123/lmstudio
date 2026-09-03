"""PHASE1 결과 그림.  판단 근거는 CSV 이고 그림은 전달용이다."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, PLOT = ROOT / "results/shared_verifier", ROOT / "plots/shared_verifier"

d = pd.read_csv(RES / "probe_results.csv")
PLOT.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
for ax, scope, title in ((axes[0], "model", "전체 (correct vs wrong)"),
                         (axes[1], "pred1", "base 가 UNSAFE 라 한 것들 중 TP vs FP")):
    g = (d[d.scope == scope].groupby(["setting", "features", "clf"]).auroc
         .agg(["mean", "std"]).reset_index())
    g["key"] = g.features + "\n" + g.clf
    order = ["conf\nlinear", "conf\nmlp", "tfidf\nlinear",
             "geom\nlinear", "geom\nmlp", "raw_last\nlinear", "raw_last\nmlp"]
    g = g[g.key.isin(order)]
    x = np.arange(len(order))
    for i, (st, c, lb) in enumerate((("per_model", "#888", "모델별 파라미터 (상한, MAIN 금지)"),
                                     ("shared", "#c0392b", "공유 파라미터 한 벌 (MAIN)"))):
        s = g[g.setting == st].set_index("key").reindex(order)
        ax.bar(x + i * .38 - .19, s["mean"], .36, yerr=s["std"], capsize=2, color=c, label=lb)
    ax.axhline(.5, color="k", ls=":", lw=1)
    for xi, k in enumerate(order):
        if k.startswith(("conf", "tfidf")):
            ax.axvspan(xi - .5, xi + .5, color="#f0e68c", alpha=.28, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylim(.4, 1.0); ax.set_ylabel("모델별 AUROC 의 평균"); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="lower left", framealpha=.9)
axes[0].text(-.4, .955, "노란 배경 = 대조군", fontsize=8)
plt.suptitle("PHASE1: 공유 검증기 vs 모델별 상한  (오차막대 = 모델 6개 간 편차; 시드 3회 변동은 ≤0.006)", fontsize=11)
plt.tight_layout(); plt.savefig(PLOT / "phase1_shared_vs_permodel.png", dpi=140); plt.close()
print(f"저장 -> {PLOT/'phase1_shared_vs_permodel.png'}")

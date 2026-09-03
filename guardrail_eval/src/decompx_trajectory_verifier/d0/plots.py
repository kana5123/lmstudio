import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import PLOTS, RES
R = RES / "phase_d0"; P = PLOTS / "phase_d0"; P.mkdir(parents=True, exist_ok=True)

lw = pd.read_csv(R / "layer_source_probe.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for c, col in (("TP", "#1f77b4"), ("FP", "#d62728")):
    s = lw[lw.cell == c]
    ax.plot(s.layer, s.macro_f1, "o-", color=col, label=f"{c}-only")
ax.axhline(1/3, ls=":", c="k", label="우연 (0.333)")
ax.set_xlabel("layer l"); ax.set_ylabel("source 3분류 macro-F1"); ax.set_ylim(0, 1.05)
ax.set_title("라벨을 고정해도 source 가 h_l 에서 복원된다"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(P / "source_probe_by_layer.png", dpi=140); plt.close()

d = pd.read_csv(R / "fp_tp_direction_cosines.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for k, lab in (("cos_WJ_PS", "WJ↔PS"), ("cos_WJ_QS", "WJ↔QS"), ("cos_PS_QS", "PS↔QS")):
    ax.plot(range(1, len(d) + 1), d[k], "o-", label=lab)
ax.axhline(0, c="k", lw=1); ax.axhline(.5, ls=":", c="gray")
ax.set_xlabel("layer l"); ax.set_ylabel("cos(d_a, d_b),  d_s = mean_FP - mean_TP")
ax.set_title("source 별 FP−TP 방향이 정렬되지 않는다"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(P / "fp_tp_direction_cosine_by_layer.png", dpi=140); plt.close()

b = pd.read_csv(R / "branch_and_occlusion_seed0.csv")
o = b[b.condition.str.startswith("occ_")]
order = [f"occ_L{i}" for i in range(1, 13)] + ["occ_early_L1-4", "occ_middle_L5-8", "occ_late_L9-12"]
piv = o.pivot(index="condition", columns="protocol", values="delta_vs_original").reindex(order)
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(order))
for i, (c, lab) in enumerate((("loso_ps", "LOSO PS"), ("loso_qs", "LOSO QS"),
                              ("loso_wj", "LOSO WJ"), ("seen_source", "seen"))):
    ax.bar(x + i * .2 - .3, piv[c], .19, label=lab)
ax.axhline(0, c="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels([s.replace("occ_", "") for s in order], rotation=45, fontsize=8)
ax.set_ylabel("ΔAUROC (개입 - 원본)"); ax.set_title("층/블록 occlusion: 9–12층만 영향을 준다")
ax.grid(axis="y", alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(P / "layer_occlusion_loso.png", dpi=140); plt.close()

cn = pd.read_csv(R / "c_norm_by_source_layer.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, t in ((axes[0], "cls_norm_median", "‖Σ_k C_lk‖ 중앙값"),
                   (axes[1], "tok_norm_median", "‖C_lk‖ 토큰 기여 중앙값")):
    p = cn.pivot_table(index="layer", columns="source_group", values=col)
    for c in p.columns:
        ax.plot(p.index, p[c], "o-", label=c.split(":")[0])
    ax.set_yscale("log"); ax.set_xlabel("layer l"); ax.set_title(t); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(P / "c_norm_by_layer_source.png", dpi=140); plt.close()

t = pd.read_csv(R / "pairwise_transfer_seed0.csv")
for mdl in ("A0", "A3"):
    p = t[t.model == mdl].pivot(index="train_source", columns="test_source", values="auroc")
    p = p.reindex(index=["WJ", "PS", "QS"], columns=["WJ", "PS", "QS"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(p.values, cmap="RdBu_r", vmin=0.2, vmax=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{p.values[i,j]:.3f}", ha="center", va="center",
                    color="white" if abs(p.values[i, j] - .6) > .25 else "black", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels(p.columns); ax.set_yticks(range(3))
    ax.set_yticklabels(p.index)
    ax.set_xlabel("평가 source"); ax.set_ylabel("학습 source")
    ax.set_title(f"{mdl} FP-vs-TP AUROC 전이 (seed 0)")
    plt.colorbar(im, ax=ax, shrink=.8)
    plt.tight_layout(); plt.savefig(P / f"pairwise_transfer_{mdl.lower()}.png", dpi=140); plt.close()
print("그림 6개 저장 ->", P)

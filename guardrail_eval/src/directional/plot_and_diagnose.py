"""층별 그림 + 보조 진단 (지시문 13·21절).

그림:
  plots/directional_alignment/layerwise_auroc.png
  plots/directional_alignment/layerwise_effect_size.png
  plots/directional_alignment/tp_fp_projection_distributions/*.png

보조 진단(‘출처 말고 다른 것으로 설명되나’):
  q 와 PG2 자체 UNSAFE 확률의 순위상관.  **q 를 만들 때 PG2 점수를 쓴 적은 없다.**
  이건 특징이 아니라 해석용 진단이다(지시문 19절의 '방향 feature 로 사용 금지'와 무관).
"""
import csv, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/directional_alignment"
ART = ROOT / "artifacts/directional_alignment"
PLOT = ROOT / "plots/directional_alignment"
FEAT = ROOT / "artifacts/features"
HELD = ("ver_dev", "eval_val", "eval_test")


def main():
    (PLOT / "tp_fp_projection_distributions").mkdir(parents=True, exist_ok=True)
    P = {s: np.load(ART / f"proj_{s}.npz", allow_pickle=True) for s in ("ver_train",) + HELD}
    L = P["eval_test"]["q"].shape[1]
    tr = [f"L{l}->L{l+1}" for l in range(L)]
    main_ix = list(range(1, L))                       # 임베딩->L1 제외

    # ---------- 층별 AUROC / 효과크기 ----------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s, st in (("ver_train", "k:"), ("ver_dev", "s--"), ("eval_val", "^--"), ("eval_test", "o-")):
        y, q = P[s]["y"], P[s]["q"]
        au = [roc_auc_score(y, q[:, li]) for li in main_ix]
        ax.plot(range(len(main_ix)), au, st, label=s, lw=2 if s == "eval_test" else 1.3)
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xticks(range(len(main_ix))); ax.set_xticklabels([tr[i] for i in main_ix], rotation=45)
    ax.set_ylabel("AUROC (TP vs FP) of q"); ax.set_xlabel("layer transition")
    ax.set_title("Layer-wise directional alignment  (direction fit on ver_train only)")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(PLOT / "layerwise_auroc.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s, st in (("ver_dev", "s--"), ("eval_val", "^--"), ("eval_test", "o-")):
        y, q = P[s]["y"], P[s]["q"]
        d = []
        for li in main_ix:
            a, b = q[y == 1, li], q[y == 0, li]
            sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
            d.append((a.mean()-b.mean())/(sp+1e-12))
        ax.plot(range(len(main_ix)), d, st, label=s)
    ax.set_xticks(range(len(main_ix))); ax.set_xticklabels([tr[i] for i in main_ix], rotation=45)
    ax.set_ylabel("Cohen's d (TP vs FP)"); ax.set_xlabel("layer transition")
    ax.set_title("Layer-wise effect size"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(PLOT / "layerwise_effect_size.png", dpi=140); plt.close()

    # ---------- 분포 그림 ----------
    for li in main_ix:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        y, q = P["eval_test"]["y"], P["eval_test"]["q"][:, li]
        ax.hist(q[y == 1], bins=60, alpha=.55, label=f"TP (n={int((y==1).sum())})", color="#c0392b")
        ax.hist(q[y == 0], bins=60, alpha=.55, label=f"FP (n={int((y==0).sum())})", color="#2471a3")
        ax.axvline(0, color="k", ls=":", label="centroid midpoint (q=0)")
        ax.set_title(f"eval_test  q  {tr[li]}   AUROC={roc_auc_score(y,q):.4f}")
        ax.set_xlabel("q = dot(v_U, g) - tau"); ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOT / "tp_fp_projection_distributions" / f"q_{tr[li].replace('->','_')}.png", dpi=130)
        plt.close()
    print(f"그림 저장 -> {PLOT}")

    # ---------- 보조 진단: q vs PG2 자체 점수 ----------
    rows = []
    print("\n=== 보조 진단: q 가 PG2 자체 확률과 얼마나 겹치나 (특징 아님, 해석용) ===")
    print(f"{'split':10} {'전이':10} {'spearman(q, p_unsafe)':>22} {'AUROC(q)':>9} {'AUROC(p_unsafe)':>16}")
    for s in HELD:
        h = torch.load(FEAT / f"hidden_{s}.pt", weights_only=False)
        idmap = {k: i for i, k in enumerate(h["sample_id"])}
        order = [idmap[k] for k in P[s]["sample_id"]]
        pu = h["unsafe_probability"].numpy()[order]
        y = P[s]["y"]
        au_p = roc_auc_score(y, pu)
        for li in main_ix:
            q = P[s]["q"][:, li]
            r = float(spearmanr(q, pu).statistic)
            rows.append({"split": s, "transition": tr[li], "spearman_q_vs_punsafe": r,
                         "auroc_q": float(roc_auc_score(y, q)), "auroc_punsafe": float(au_p)})
            if s == "eval_test":
                print(f"{s:10} {tr[li]:10} {r:22.4f} {roc_auc_score(y,q):9.4f} {au_p:16.4f}")
    with open(RES / "q_vs_pg2_score_diagnostic.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"저장 -> {RES/'q_vs_pg2_score_diagnostic.csv'}")


if __name__ == "__main__":
    main()

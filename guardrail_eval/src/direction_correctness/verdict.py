"""§18 판정 — 층마다 Case A 조건을 모두 만족하는지 체계적으로 집계.

'최고 AUROC 층'만 골라 보면 선택 편향이 생기므로, 전 층에 대해 아래 5개 조건을
각각 판정하고 몇 개 층이 통과하는지 센다.

  C1 held-out correctness AUROC 의 부트스트랩 95% CI 가 0.5 를 넘음
  C2 두 부분집합(TP vs FP, TN vs FN)이 **둘 다** 0.5 초과 (같은 부호)
  C3 라벨 누수 낮음: |AUROC(delta_CORR -> attack/benign) - 0.5| <= 0.10
  C4 seed 간 방향 안정: 평균 cos >= 0.8
  C5 치환 귀무 대비 유의: empirical p <= 0.05
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/direction_correctness"
LEAK_TOL = 0.10
COS_MIN = 0.80
P_MAX = 0.05


def main(tag="move"):
    A = pd.read_csv(RES / f"correctness_auroc_by_layer__{tag}.csv")
    S = pd.read_csv(RES / f"subgroup_auroc_by_layer__{tag}.csv")
    L = pd.read_csv(RES / f"label_leakage_by_layer__{tag}.csv")
    Rp = pd.read_csv(RES / f"split_reproducibility_by_layer__{tag}.csv")
    Pn = pd.read_csv(RES / f"permutation_null_by_layer__{tag}.csv")

    a = A.groupby(["dataset", "layer"]).agg(auroc=("auroc_correctness", "mean"),
                                            ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean")).reset_index()
    s = S.groupby(["dataset", "layer"])[["auroc_danger_TPvFP", "auroc_safe_TNvFN"]].mean().reset_index()
    l = L.groupby(["dataset", "layer"])["CORR_auroc_label"].mean().reset_index()
    r = Rp[Rp.direction == "CORR"].groupby(["dataset", "layer"])["mean"].mean().reset_index() \
        .rename(columns={"mean": "seed_cos"})
    p = Pn.groupby(["dataset", "layer"])["empirical_p"].mean().reset_index()
    m = a.merge(s, on=["dataset", "layer"]).merge(l, on=["dataset", "layer"]) \
         .merge(r, on=["dataset", "layer"], how="left").merge(p, on=["dataset", "layer"], how="left")
    m = m[~m.layer.isin(["L0->L1", "L0"])]
    m["C1_ci_above_half"] = m.ci_lo > 0.5
    m["C2_both_subgroups"] = (m.auroc_danger_TPvFP > 0.5) & (m.auroc_safe_TNvFN > 0.5)
    m["C3_low_label_leak"] = (m.CORR_auroc_label - 0.5).abs() <= LEAK_TOL
    m["C4_split_stable"] = m.seed_cos >= COS_MIN
    m["C5_perm_sig"] = m.empirical_p <= P_MAX
    m["n_pass"] = m[[f"C{i}_" + x for i, x in enumerate(
        ["ci_above_half", "both_subgroups", "low_label_leak", "split_stable", "perm_sig"], 1)]].sum(1)
    m["ALL5"] = m["n_pass"] == 5
    m.to_csv(RES / f"verdict_by_layer__{tag}.csv", index=False)

    print(f"########## repr={tag}  (층당 5개 조건) ##########")
    print(f"{'dataset':20} {'층수':>4} {'C1':>3} {'C2':>3} {'C3':>3} {'C4':>3} {'C5':>3} {'전부통과':>8}"
          f"  {'최고AUROC':>9} {'그층 라벨누수':>11}")
    for ds, g in m.groupby("dataset"):
        b = g.loc[g.auroc.idxmax()]
        print(f"{ds:20} {len(g):4} "
              f"{int(g.C1_ci_above_half.sum()):3} {int(g.C2_both_subgroups.sum()):3} "
              f"{int(g.C3_low_label_leak.sum()):3} {int(g.C4_split_stable.sum()):3} "
              f"{int(g.C5_perm_sig.sum()):3} {int(g.ALL5.sum()):8}"
              f"  {b.auroc:9.3f} {b.CORR_auroc_label:11.3f}")
    print()
    print("=== 5개 조건을 모두 통과한 (데이터셋, 층) ===")
    ok = m[m.ALL5]
    if ok.empty:
        print("  없음")
    else:
        for _, x in ok.iterrows():
            print(f"  {x.dataset:20} {x.layer:10} AUROC={x.auroc:.3f} "
                  f"[{x.ci_lo:.3f},{x.ci_hi:.3f}]  TPvFP={x.auroc_danger_TPvFP:.3f} "
                  f"TNvFN={x.auroc_safe_TNvFN:.3f}  라벨={x.CORR_auroc_label:.3f} "
                  f"cos={x.seed_cos:.3f} p={x.empirical_p:.4f}")
    print(f"\n저장 -> {RES/f'verdict_by_layer__{tag}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "move")

import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.base_ood_audit import (OUT, curve_points, fpr_resolution,
                                                            load_population, low_fpr,
                                                            score_distribution, source_metrics)
from src.decompx_trajectory_verifier.config import RES
pd.set_option("display.width", 230)
OUT.mkdir(parents=True, exist_ok=True)
df = load_population()
sm = source_metrics(df);        sm.to_csv(OUT / "base_source_metrics.csv", index=False)
lf = low_fpr(df);               lf.to_csv(OUT / "base_low_fpr_metrics.csv", index=False)
sd = score_distribution(df);    sd.to_csv(OUT / "base_score_distribution.csv", index=False)
roc, pr = curve_points(df)
roc.to_csv(OUT / "base_roc_points.csv", index=False); pr.to_csv(OUT / "base_pr_points.csv", index=False)
fr = fpr_resolution(df);        fr.to_csv(OUT / "base_fpr_resolution.csv", index=False)

# --- C1 verifier 결과와 나란히 (곱하거나 더하지 않는다) ---------------------
runs = pd.read_csv(RES / "phase_c1/training_runs.csv")
FOLD = {"loso_wj": "wildjailbreak:adversarial", "loso_ps": "promptshield:test",
        "loso_qs": "piguard:Question Set"}
v = runs[runs.protocol.isin(FOLD)].copy()
v["source_group"] = v.protocol.map(FOLD)
piv = v.pivot_table(index="source_group", columns="model", values="test_auroc", aggfunc="mean")
r1 = lf[(lf.target_fpr == 0.01) & lf.feasible].set_index("source_group").achieved_recall
summary = (sm.set_index("source_group")[["num_benign", "num_attack", "base_auroc", "base_auprc",
                                         "native_recall", "native_fpr"]]
           .join(r1.rename("base_recall_at_1pct_fpr")).join(piv.add_prefix("verifier_loso_")))
summary.to_csv(OUT / "base_vs_verifier_summary.csv")

print("=== 1-2. source별 population 과 혼동셀 ===")
print(sm[["source_group", "num_benign", "num_attack", "n", "TP", "FP", "TN", "FN"]].to_string(index=False))
print("\n=== 3-4. Base AUROC/AUPRC 와 native threshold ===")
print(sm[["source_group", "base_auroc", "base_auprc", "native_recall", "native_fpr",
          "native_precision", "native_specificity"]].round(4).to_string(index=False))
print("   native recall CI / fpr CI:")
print(sm[["source_group","native_recall_ci_lo","native_recall_ci_hi","native_fpr_ci_lo","native_fpr_ci_hi"]].round(5).to_string(index=False))
print("\n=== 5. Recall @ FPR 제약 (제약 만족 중 recall 최대 threshold) ===")
print(lf[["source_group", "target_fpr", "threshold", "achieved_fpr", "achieved_recall",
          "tp_count", "fp_count", "benign_denominator", "attack_denominator",
          "recall_ci_lo", "recall_ci_hi", "statistically_low_resolution"]].round(5).to_string(index=False))
print("\n=== 6. FPR 통계 해상도 ===")
print(fr.round(6).to_string(index=False))
print("\n=== 7. base score(z_attack - z_benign) 분포 ===")
print(sd.round(3).to_string(index=False))
print("\n=== 8. Base 와 C1 verifier 나란히 (합성하지 않음) ===")
print(summary.round(4).to_string())

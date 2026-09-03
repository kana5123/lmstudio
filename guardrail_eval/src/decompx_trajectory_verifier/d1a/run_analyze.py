import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import RES
from src.decompx_trajectory_verifier.d1a.analyze import OUT, bootstrap, low_tp_loss, results
OUT.mkdir(parents=True, exist_ok=True)
nat = results(False); nat.to_csv(OUT / "natural_results.csv", index=False)
nat[nat.protocol == "seen_source"].to_csv(OUT / "natural_seen_results.csv", index=False)
nat[nat.protocol != "seen_source"].to_csv(OUT / "natural_loso_results.csv", index=False)
mat = results(True); mat.to_csv(OUT / "matched_results.csv", index=False)
mat[mat.protocol == "seen_source"].to_csv(OUT / "matched_seen_results.csv", index=False)
mat[mat.protocol != "seen_source"].to_csv(OUT / "matched_loso_results.csv", index=False)
low_tp_loss().to_csv(OUT / "low_tp_loss.csv", index=False)
bootstrap(False).to_csv(OUT / "paired_bootstrap_natural.csv", index=False)
bootstrap(True).to_csv(OUT / "paired_bootstrap_matched.csv", index=False)
runs = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("runs_*.csv"))], ignore_index=True)
runs.to_csv(OUT / "training_runs.csv", index=False)
runs.groupby(["protocol", "variant"]).agg(
    n=("seed", "count"), auroc_mean=("test_auroc", "mean"), auroc_std=("test_auroc", "std"),
    auroc_min=("test_auroc", "min"), auroc_max=("test_auroc", "max"),
    auprc_mean=("test_auprc", "mean"), auprc_std=("test_auprc", "std"),
    ep_mean=("best_epoch", "mean"), minutes=("minutes", "mean")).round(4).to_csv(
    OUT / "seed_stability.csv")
print("집계 완료", len(nat), len(mat))

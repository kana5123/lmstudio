import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.analyze_c1 import (audit_flag_sensitivity, collect,
                                                        low_tp_loss, paired, seed_stability)
from src.decompx_trajectory_verifier.config import RES
OUT = RES / "phase_c1"
P = ["seen_source", "loso_wj", "loso_ps", "loso_qs"]; M = ["M0", "A0", "A3"]; S = [0, 1, 2, 3, 4]
runs = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("training_runs_seed*.csv"))],
                 ignore_index=True)
runs.to_csv(OUT / "training_runs.csv", index=False)
pooled, per_src = collect(P, M, S)
pooled.to_csv(OUT / "seen_source_results.csv", index=False)
per_src.to_csv(OUT / "seen_source_per_dataset.csv", index=False)
pooled[pooled.protocol != "seen_source"].to_csv(OUT / "loso_results.csv", index=False)
low_tp_loss(P, M, S).to_csv(OUT / "low_tp_loss_metrics.csv", index=False)
paired(P, S).to_csv(OUT / "paired_bootstrap.csv", index=False)
seed_stability(runs).to_csv(OUT / "seed_stability.csv")
audit_flag_sensitivity(P, M, S).to_csv(OUT / "audit_flag_sensitivity.csv", index=False)
print("집계 완료")

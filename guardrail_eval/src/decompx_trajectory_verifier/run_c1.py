"""PHASE C1 실행: protocol x model x seed."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.metrics import core_metrics
from src.decompx_trajectory_verifier.train_c1 import run

SPLITDIR = ART / "phase_c1/split_manifests"
OUT = ART / "phase_c1"
PROTOCOLS = ["seen_source", "loso_wj", "loso_ps", "loso_qs"]
MODELS = ["M0", "A0", "A3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()

    for d in ("checkpoints", "scalers", "prediction_files"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    (RES / "phase_c1").mkdir(parents=True, exist_ok=True)
    rows = []
    for proto in a.protocols.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            man = pd.read_parquet(SPLITDIR / f"{proto}_seed{seed}.parquet")
            for mdl in a.models.split(","):
                key = f"{proto}_{mdl}_seed{seed}"
                t0 = time.time()
                print(f"\n=== {key} ===", flush=True)
                r = run(mdl, man, seed, dev=a.dev, bs=a.bs,
                        log=lambda s: print(s, flush=True))
                torch.save(r["model"].state_dict(), OUT / f"checkpoints/{key}.pt")
                sc = {k: v.tolist() for k, v in r["model"].state_dict().items()
                      if k.endswith(("mean", "std"))}
                json.dump(sc, open(OUT / f"scalers/{key}.json", "w"))
                for sp, df in r["preds"].items():
                    df.to_parquet(OUT / f"prediction_files/{key}_{sp}.parquet", index=False)
                r["history"].to_csv(RES / f"phase_c1/history_{key}.csv", index=False)
                te = r["preds"]["test"]
                m = core_metrics(te.y_fp.to_numpy(), te.p_fp.to_numpy())
                rows.append(dict(protocol=proto, model=mdl, seed=seed,
                                 n_params=r["n_params"], best_epoch=r["best_epoch"],
                                 best_val_macro_auprc=r["best_val_macro_auprc"],
                                 a_max_dev=r["a_max_dev"], minutes=(time.time() - t0) / 60,
                                 **{f"test_{k}": v for k, v in m.items()}))
                print(f"  -> test AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} "
                      f"({(time.time()-t0)/60:.1f}분)", flush=True)
                del r
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    p = RES / f"phase_c1/training_runs_seed{a.seeds.replace(',','-')}.csv"
    df.to_csv(p, index=False)
    print(f"\n저장 -> {p}")
    print(df[["protocol", "model", "seed", "best_epoch", "test_auroc", "test_auprc",
              "minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()

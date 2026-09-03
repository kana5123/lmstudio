"""PHASE D1A 학습: V0~V4 x protocol x seed.  split/설정은 C1 과 완전히 동일."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.metrics import core_metrics
from src.decompx_trajectory_verifier.train_c1 import run

SPLITS = ART / "phase_c1/split_manifests"
OUT = ART / "phase_d1a"
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    for d in ("checkpoints", "predictions"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    (RES / "phase_d1a").mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(SPLITS / f"{a.protocol}_seed{a.seed}.parquet")
    rows = []
    for v in a.variants.split(","):
        key = f"{a.protocol}_{v}_seed{a.seed}"
        t0 = time.time()
        print(f"\n=== {key} ===", flush=True)
        r = run(v, man, a.seed, bs=a.bs, log=lambda s: print(s, flush=True))
        torch.save(r["model"].state_dict(), OUT / f"checkpoints/{key}.pt")
        for sp, df in r["preds"].items():
            df.to_parquet(OUT / f"predictions/{key}_{sp}.parquet", index=False)
        r["history"].to_csv(RES / f"phase_d1a/history_{key}.csv", index=False)
        te = r["preds"]["test"]
        m = core_metrics(te.y_fp.to_numpy(), te.p_fp.to_numpy())
        rows.append(dict(protocol=a.protocol, variant=v, seed=a.seed, n_params=r["n_params"],
                         best_epoch=r["best_epoch"], best_val_macro_auprc=r["best_val_macro_auprc"],
                         minutes=(time.time() - t0) / 60,
                         **{f"test_{k}": val for k, val in m.items()}))
        print(f"  -> test AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} "
              f"({(time.time()-t0)/60:.1f}분)", flush=True)
        del r
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(RES / f"phase_d1a/runs_{a.protocol}_seed{a.seed}.csv", index=False)
    print("완료", flush=True)


if __name__ == "__main__":
    main()

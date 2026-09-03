"""§19 cached verifier latency.

캐시된 C/Y evidence 를 입력으로 하는 A3 verifier 자체의 지연만 잰다.
PromptGuard2 + DecompX 전체 지연이 아니다.  end-to-end production latency 라고 부르지 않는다.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.data import MemmapEvidence
from src.decompx_trajectory_verifier.model import DXTV

BUCKETS = [(1, 64), (65, 128), (129, 256), (257, 512)]


@torch.no_grad()
def main(ckpt=None, dev="cuda", n_per_bucket=50, warmup=10):
    mm = MemmapEvidence()
    idx = mm.index.reset_index()
    m = DXTV(mm.d, mm.L, 512, variant="A3", d_v=C.D_V, depth_tf=C.DEPTH_TF, token_tf=C.TOKEN_TF,
             attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT, head_hidden=C.HEAD_HIDDEN).to(dev)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in BUCKETS:
        pool = idx[(idx.length >= lo) & (idx.length <= hi)]
        if not len(pool):
            continue
        pick = pool.iloc[rng.choice(len(pool), size=min(n_per_bucket + warmup, len(pool)),
                                    replace=False)]
        ts = []
        torch.cuda.reset_peak_memory_stats(dev)
        for k, (_, r) in enumerate(pick.iterrows()):
            o, n = int(r.offset), int(r.length)
            Cx = torch.from_numpy(np.array(mm.C[o:o + n])).permute(1, 0, 2)[None].to(dev)
            Y = torch.from_numpy(np.array(mm.Y[o:o + n])).to(dev)
            fa = torch.stack([Y[:, 0], Y[:, 1], Y[:, 1] - Y[:, 0]], -1)[None]
            mk = torch.ones(1, n, device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(Cx, fa, mk, None)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if k >= warmup:
                ts.append(dt)
        peak = torch.cuda.max_memory_allocated(dev) / 1e6
        ts = np.array(ts)
        rows.append(dict(token_bucket=f"{lo}-{hi}", n=len(ts), batch_size=1,
                         p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)),
                         mean_ms=float(ts.mean()), peak_gpu_mem_MB=peak,
                         measurement="cached verifier latency (A3 only, evidence precomputed)"))
    df = pd.DataFrame(rows)
    (RES / "phase_c1").mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "phase_c1/verifier_only_latency.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=None)
    main(**vars(ap.parse_args()))

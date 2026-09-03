"""PHASE D0 공통 표현 캐시.

h_l = sum_k C_lk  (DecompX 계약상 layer l 의 실제 CLS hidden 과 같다)
그리고 토큰 기여 노름 ||C_lk||_2 의 표본별 통계.
새 추출은 하지 않는다.  B3 fp32 memmap 만 읽는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import ART, DATA
from src.decompx_trajectory_verifier.data import MemmapEvidence

OUT = ART / "phase_d0"


def build(batch=256):
    OUT.mkdir(parents=True, exist_ok=True)
    mm = MemmapEvidence()
    core = pd.read_parquet(DATA / "core_tp_fp.parquet")
    j = core.join(mm.index[["offset", "length"]], on="sample_id")
    n, L, d = len(j), mm.L, mm.d
    H = np.zeros((n, L, d), dtype=np.float32)          # h_l = sum_k C_lk
    Ymean = np.zeros((n, 3), dtype=np.float32)         # [Y_B, Y_A, a] 의 표본 평균
    tok_norm_stats = np.zeros((n, L, 5), dtype=np.float32)   # median,p25,p75,p95,mean of ||C_lk||
    off = j.offset.to_numpy(); ln = j.length.to_numpy()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        for i in range(s, e):
            o, t = int(off[i]), int(ln[i])
            C = np.asarray(mm.C[o:o + t])              # [T,L,d]
            H[i] = C.sum(0)
            nr = np.linalg.norm(C, axis=-1)            # [T,L]
            tok_norm_stats[i, :, 0] = np.median(nr, 0)
            tok_norm_stats[i, :, 1] = np.percentile(nr, 25, axis=0)
            tok_norm_stats[i, :, 2] = np.percentile(nr, 75, axis=0)
            tok_norm_stats[i, :, 3] = np.percentile(nr, 95, axis=0)
            tok_norm_stats[i, :, 4] = nr.mean(0)
            Yb = np.asarray(mm.Y[o:o + t])
            Ymean[i] = [Yb[:, 0].mean(), Yb[:, 1].mean(), (Yb[:, 1] - Yb[:, 0]).mean()]
        if (s // batch) % 10 == 0:
            print(f"  {e:,}/{n:,}", flush=True)
    np.save(OUT / "h_layers.npy", H)
    np.save(OUT / "tok_norm_stats.npy", tok_norm_stats)
    np.save(OUT / "y_mean.npy", Ymean)
    meta = j[["sample_id", "source_group", "source_subgroup", "confusion_cell",
              "duplicate_group_id", "token_length", "split"]].reset_index(drop=True)
    meta["y_fp"] = (meta.confusion_cell == "FP").astype(np.int8)
    meta.to_parquet(OUT / "meta.parquet", index=False)
    print(f"저장 -> {OUT}  H {H.shape}  norm_stats {tok_norm_stats.shape}")


if __name__ == "__main__":
    build()

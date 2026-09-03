"""§18: 학습 I/O 를 위해 청크 캐시를 memmap 형식으로 변환한다.

전체를 RAM 에 올리지 않고 numpy memmap 으로 필요한 구간만 읽는다.
fp32 원본을 그대로 옮기며 정밀도 변환은 하지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import ART

CACHE = ART / "core_cache"
MM = ART / "memmap"


def main():
    files = sorted(CACHE.glob("core_s*_c*.pt"))
    MM.mkdir(parents=True, exist_ok=True)
    # 1차 통과: 총 토큰 수와 메타데이터
    rows, total = [], 0
    for f in files:
        b = torch.load(f, weights_only=False, map_location="cpu")
        off = b["offsets"].numpy()
        L, d = b["L"], b["d"]
        for i, sid in enumerate(b["sample_id"]):
            n = int(off[i + 1] - off[i])
            rows.append(dict(sample_id=sid, offset=total, length=n,
                             logit_0=float(b["logits"][i, 0]), logit_1=float(b["logits"][i, 1]),
                             source_group=b["source_group"][i],
                             source_subgroup=b["source_subgroup"][i],
                             confusion_cell=b["confusion_cell"][i],
                             duplicate_group_id=b["duplicate_group_id"][i],
                             chunk=f.name, idx_in_chunk=i))
            total += n
        del b
    idx = pd.DataFrame(rows)
    print(f"표본 {len(idx):,}  총 토큰 {total:,}  L={L} d={d}", flush=True)

    C = np.lib.format.open_memmap(MM / "C.npy", mode="w+", dtype=np.float32, shape=(total, L, d))
    Y = np.lib.format.open_memmap(MM / "Y.npy", mode="w+", dtype=np.float32, shape=(total, 2))
    A = np.lib.format.open_memmap(MM / "a.npy", mode="w+", dtype=np.float32, shape=(total,))
    I = np.lib.format.open_memmap(MM / "ids.npy", mode="w+", dtype=np.int32, shape=(total,))

    pos = 0
    for k, f in enumerate(files):
        b = torch.load(f, weights_only=False, map_location="cpu")
        n = b["C_flat"].shape[0]
        C[pos:pos + n] = b["C_flat"].numpy()
        Y[pos:pos + n] = b["Y_flat"].numpy()
        A[pos:pos + n] = b["a_flat"].numpy()
        I[pos:pos + n] = b["ids_flat"].numpy()
        pos += n
        del b
        print(f"  [{k+1}/{len(files)}] {f.name}  누적 토큰 {pos:,}", flush=True)
    assert pos == total, f"{pos} != {total}"
    C.flush(); Y.flush(); A.flush(); I.flush()
    idx.to_parquet(MM / "index.parquet", index=False)
    meta = dict(total_tokens=int(total), L=int(L), d=int(d), n_samples=len(idx))
    pd.Series(meta).to_json(MM / "meta.json")
    print(f"완료 -> {MM}", flush=True)


if __name__ == "__main__":
    main()

"""STEP 14: 전체 evidence 캐시 생성.  분할별로 샤딩해 여러 GPU 에 나눠 돌린다."""
import argparse, sys, time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_verifier_v2.config import ART, DATA
from src.decompx_verifier_v2.evidence_cache import EvidenceExtractor, extract_to_file

CHUNK = 2000     # 샤드 파일 하나당 표본 수


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()

    d = pd.read_parquet(DATA / "samples.parquet")
    d = d[d.split.isin(a.splits.split(","))].reset_index(drop=True)
    # 길이순으로 정렬한 뒤 라운드로빈 -> 샤드마다 길이 분포가 비슷해진다
    d = d.sort_values("token_length").reset_index(drop=True)
    d = d[d.index % a.nshards == a.shard].reset_index(drop=True)
    print(f"샤드 {a.shard}/{a.nshards}: 표본 {len(d):,}", flush=True)

    ex = EvidenceExtractor()
    outdir = ART / "cache"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ci, s in enumerate(range(0, len(d), CHUNK)):
        p = outdir / f"ev_s{a.shard}_c{ci:03d}.pt"
        if p.exists():
            continue
        sub = d.iloc[s:s + CHUNK].reset_index(drop=True)
        extract_to_file(sub, p, ex=ex)
        el = time.time() - t0
        done = s + len(sub)
        print(f"  [{a.shard}] {done:,}/{len(d):,}  {el/60:.1f}분 경과  "
              f"남은 예상 {el/done*(len(d)-done)/60:.1f}분", flush=True)
    print(f"샤드 {a.shard} 완료 {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

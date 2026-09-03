"""PHASE B3: CORE 11,947 건 fp32 evidence 추출.  샤드별로 GPU 하나씩 쓴다.

각 표본은 정확히 하나의 샤드에만 들어간다(길이순 정렬 후 라운드로빈).
샤드마다 manifest 와 계약 감사 CSV 를 남긴다.
"""
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

from src.decompx_trajectory_verifier.config import ART, DATA, RES
from src.decompx_trajectory_verifier.evidence_cache import EvidenceExtractor, save_shard

CHUNK = 250          # 청크 파일 하나당 표본 수 (fp32 기준 파일 약 2 GB)


def audit_rows(blob, chunk_id):
    rows_e, rows_h, rows_m, rows_p = [], [], [], []
    for i, sid in enumerate(blob["sample_id"]):
        a = blob["audits"][i]
        meta = dict(sample_id=sid, source_group=blob["source_group"][i],
                    source_subgroup=blob["source_subgroup"][i], chunk=chunk_id)
        for l in range(len(a["enc_abs"])):
            rows_e.append(dict(**meta, layer=l + 1, absolute_error=float(a["enc_abs"][l]),
                               normalized_error=float(a["enc_norm"][l]),
                               max_coordinate_error=float(a["enc_maxc"][l])))
        for c in range(len(a["head_abs"])):
            rows_h.append(dict(**meta, class_id=c, absolute_error=float(a["head_abs"][c]),
                               normalized_error=float(a["head_norm"][c]),
                               logit_norm=a["logit_norm"]))
        rows_m.append(dict(**meta, absolute_error=a["mar_abs"], normalized_error=a["mar_norm"]))
        rows_p.append(dict(**meta, pad_C_max=a["padC"], pad_Y_max=a["padY"], pad_a_max=a["padA"]))
    return rows_e, rows_h, rows_m, rows_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    core = pd.read_parquet(DATA / "core_tp_fp.parquet")
    core = core.sort_values(["token_length", "sample_id"]).reset_index(drop=True)
    mine = core[core.index % a.nshards == a.shard].reset_index(drop=True)
    outdir = ART / "core_cache"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"샤드 {a.shard}/{a.nshards} gpu={a.gpu}: 표본 {len(mine):,} "
          f"(TP {int((mine.confusion_cell=='TP').sum()):,} / FP {int((mine.confusion_cell=='FP').sum()):,})",
          flush=True)

    ex = EvidenceExtractor(store_dtype=torch.float32)
    E, H, M, P = [], [], [], []
    t0, ntok = time.time(), 0
    for ci, s in enumerate(range(0, len(mine), CHUNK)):
        p = outdir / f"core_s{a.shard}_c{ci:03d}.pt"
        sub = mine.iloc[s:s + CHUNK].reset_index(drop=True)
        if p.exists():
            blob = torch.load(p, weights_only=False)
        else:
            blob = save_shard(sub, p, ex)
        e, h, m, pd_ = audit_rows(blob, ci)
        E += e; H += h; M += m; P += pd_
        ntok += int(blob["offsets"][-1])
        done = s + len(sub)
        el = time.time() - t0
        print(f"  [{a.shard}] {done:,}/{len(mine):,}  토큰 {ntok:,}  {el/60:.1f}분  "
              f"남은 예상 {el/done*(len(mine)-done)/60:.1f}분", flush=True)
        del blob

    RES.mkdir(parents=True, exist_ok=True)
    for nm, rows in (("encoder", E), ("head", H), ("margin", M), ("padding", P)):
        pd.DataFrame(rows).to_csv(RES / f"b3_{nm}_audit_s{a.shard}.csv", index=False)
    man = dict(shard_id=a.shard, gpu_id=a.gpu, num_samples=len(mine), num_tokens=ntok,
               TP=int((mine.confusion_cell == "TP").sum()),
               FP=int((mine.confusion_cell == "FP").sum()),
               by_source_group=mine.source_group.value_counts().to_dict(),
               by_source_subgroup=mine.source_subgroup.value_counts().to_dict(),
               first_sample_id=mine.sample_id.iloc[0], last_sample_id=mine.sample_id.iloc[-1],
               n_chunks=int(np.ceil(len(mine) / CHUNK)),
               elapsed_min=round((time.time() - t0) / 60, 2))
    json.dump(man, open(RES / f"b3_manifest_s{a.shard}.json", "w"), indent=1, ensure_ascii=False)
    print(f"샤드 {a.shard} 완료 {man['elapsed_min']}분, 토큰 {ntok:,}", flush=True)


if __name__ == "__main__":
    main()

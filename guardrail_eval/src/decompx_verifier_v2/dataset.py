"""PHASE A 에서 확정된 MAIN 중 심사 학회 게재 출처만 사용한다(§28).

데이터셋 선정을 다시 분석하지 않는다.  기존 group_id 로 group-aware 분할만 만든다.
같은 중복 그룹은 한 분할에만 존재한다.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier_v2.config import DATA, PEER_REVIEWED_SOURCES

PRED = Path(__file__).resolve().parents[2] / "data/decompx_verifier/pg2_predictions.parquet"
FRAC = dict(train=0.70, val=0.15, test=0.15)


def split_of(group_id, seed=0):
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{group_id}".encode(), digest_size=8).digest(),
                       "big") / 2 ** 64
    return "train" if h < FRAC["train"] else ("val" if h < FRAC["train"] + FRAC["val"] else "test")


def build(seed=0, save=True):
    d = pd.read_parquet(PRED)
    d = d[(d.use == "MAIN") & d.length_ok].copy()
    d["source"] = d.dataset.str.split(":").str[-1]
    d = d[d.source.isin(PEER_REVIEWED_SOURCES)].reset_index(drop=True)
    d["split"] = [split_of(g, seed) for g in d.group_id]
    bad = d.groupby("group_id").split.nunique()
    assert (bad == 1).all(), "그룹이 분할에 걸쳐 있다"
    keep = ["sample_id", "text", "dataset", "source", "group_id", "confusion_cell",
            "base_pred", "gt", "p_unsafe", "token_length", "split"]
    d = d[keep]
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        d.to_parquet(DATA / "samples.parquet", index=False)
    return d


if __name__ == "__main__":
    d = build()
    print(f"심사 게재 MAIN 표본 {len(d):,}  출처 {d.source.nunique()}개  "
          f"그룹 {d.group_id.nunique():,}")
    print()
    t = d.pivot_table(index="split", columns="confusion_cell", values="sample_id",
                      aggfunc="count", fill_value=0).reindex(["train", "val", "test"])
    t = t.reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0)
    t["attack분기(TP+FP)"] = t.TP + t.FP
    t["benign분기(TN+FN)"] = t.TN + t.FN
    t["합"] = t[["TP", "FP", "TN", "FN"]].sum(1)
    print(t.to_string())
    print()
    print("출처별 셀 (상위)")
    s = d.pivot_table(index="source", columns="confusion_cell", values="sample_id",
                      aggfunc="count", fill_value=0).reindex(columns=["TP","FP","TN","FN"], fill_value=0)
    print(s.assign(합=s.sum(1)).sort_values("합", ascending=False).to_string())

"""CORE TP/FP 집합 선정과 group-aware 분할 (§4, §5, §25).

§5: 같은 source_group 안에 TP 와 FP 가 모두 존재하는 source 만 MAIN 학습에 쓴다.
    한쪽 cell 만 있는 source 는 stress-test metadata 로만 보존한다.
§25: 같은 duplicate_group_id 는 한 분할에만 들어간다.
     그리고 각 source 안에서 train/val/test 가 모두 TP 와 FP 를 포함해야 한다.

source 이름과 개수는 기존 PHASE A 산출물에서 읽는다.  새 source 분석은 하지 않는다.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import DATA

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data/decompx_verifier/pg2_predictions.parquet"
CANON = ROOT / "data/multisource_guard/canonical_samples.parquet"
MIN_CELL = 30                       # 각 분할이 두 cell 을 다 갖도록 하는 최소치
FRAC = dict(train=0.70, val=0.15, test=0.15)


def _hash01(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), "big") / 2**64


def load_pool():
    """PHASE A MAIN 중 base 가 ATTACK 이라 예측한 것(TP/FP)만."""
    c = pd.read_parquet(CANON, columns=["sample_id", "source_group", "original_split"])
    p = pd.read_parquet(PRED)
    p = p[(p.use == "MAIN") & p.length_ok].merge(c, on="sample_id", how="left")
    p = p.rename(columns={"group_id": "duplicate_group_id"})
    return p[p.confusion_cell.isin(["TP", "FP"])].reset_index(drop=True)


def usable_sources(pool, min_cell=MIN_CELL):
    g = (pool.pivot_table(index="source_group", columns="confusion_cell",
                          values="sample_id", aggfunc="count", fill_value=0)
         .reindex(columns=["TP", "FP"], fill_value=0))
    return g, list(g[g.min(axis=1) >= min_cell].index)


def assign_splits(df, seed=0):
    """(source_group, cell) 층 안에서 중복 그룹 단위로 70/15/15 배정.

    층별로 나누므로 각 source 의 각 분할에 TP 와 FP 가 모두 들어간다.
    중복 그룹이 두 cell 에 걸치면 처음 배정을 따르고, 걸침 여부는 assert 로 확인한다."""
    out = {}
    for (src, cell), sub in df.groupby(["source_group", "confusion_cell"]):
        groups = sorted(sub.duplicate_group_id.unique(),
                        key=lambda g: _hash01(f"{seed}:{src}:{cell}:{g}"))
        n = len(groups)
        n_tr, n_va = int(round(n * FRAC["train"])), int(round(n * FRAC["val"]))
        for i, g in enumerate(groups):
            if g in out:
                continue                      # 다른 cell 에서 이미 배정됨 -> 유지
            out[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    s = df.duplicate_group_id.map(out)
    assert s.notna().all()
    return s


def build(seed=0, save=True):
    pool = load_pool()
    counts, usable = usable_sources(pool)
    core = pool[pool.source_group.isin(usable)].copy()
    core["split"] = assign_splits(core, seed)
    bad = core.groupby("duplicate_group_id").split.nunique()
    assert (bad == 1).all(), f"중복 그룹 {int((bad>1).sum())}개가 분할에 걸침"
    core["y_fp"] = (core.confusion_cell == "FP").astype(np.int8)
    # §1: adversarial_benign / adversarial_harmful 은 같은 source_group 아래의
    #     세부 provenance 다.  training 단위는 source_group, 기록은 둘 다 남긴다.
    core["source_subgroup"] = core["original_source"]
    keep = ["sample_id", "text", "dataset", "source_group", "source_subgroup",
            "original_source", "duplicate_group_id", "original_split",
            "confusion_cell", "y_fp", "base_pred", "gt", "p_unsafe",
            "token_length", "split"]
    core = core[[k for k in keep if k in core.columns]].reset_index(drop=True)
    stress = pool[~pool.source_group.isin(usable)][
        ["sample_id", "source_group", "confusion_cell", "token_length"]].reset_index(drop=True)
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        core.to_parquet(DATA / "core_tp_fp.parquet", index=False)
        stress.to_parquet(DATA / "stress_only_one_cell.parquet", index=False)
        counts.to_csv(DATA / "source_cell_counts.csv")
    return core, stress, counts, usable


if __name__ == "__main__":
    core, stress, counts, usable = build()
    print(f"§5 사용 가능 source_group {len(usable)}개 (TP·FP 각 {MIN_CELL}개 이상)")
    print(counts.loc[usable].assign(합=lambda x: x.TP + x.FP).to_string())
    print(f"\nCORE {len(core):,}건  |  한쪽 cell 만 있어 제외 {len(stress):,}건")
    t = core.pivot_table(index=["source_group", "split"], columns="confusion_cell",
                         values="sample_id", aggfunc="count", fill_value=0)
    t = t.reindex(columns=["TP", "FP"], fill_value=0)
    t["FP비율"] = (t.FP / (t.TP + t.FP)).round(3)
    print("\n=== source x split (§25: 모든 칸에 TP·FP 둘 다 있어야 함) ===")
    print(t.to_string())
    assert (t.TP > 0).all() and (t.FP > 0).all(), "★ 어떤 분할에 한 cell 이 비었다"
    print("\n모든 (source, split) 칸에 TP·FP 둘 다 존재 -> §25 통과")

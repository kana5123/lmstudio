"""PHASE E0: 전체 population 복원, group-aware 분할, base score 표."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.high_recall_attribution_cascade.config import (DATA, EXCLUDED_UNREVIEWED, RES,
                                                        SOURCE_MAP, SPLIT_FRACS)

ROOT = Path(__file__).resolve().parents[2]


def load_population():
    # pg2_predictions 에 이미 original_source 가 있으므로 canonical 에서는 source_group 만 가져온다
    c = pd.read_parquet(ROOT / "data/multisource_guard/canonical_samples.parquet",
                        columns=["sample_id", "source_group"])
    p = pd.read_parquet(ROOT / "data/decompx_verifier/pg2_predictions.parquet")
    p = p[(p.use == "MAIN") & p.length_ok].merge(c, on="sample_id", how="left")
    p = p.rename(columns={"gt": "gt_attack", "group_id": "duplicate_group_id",
                          "source_group": "source_group_raw",
                          "original_source": "source_subgroup"})
    p["source_group"] = p.source_group_raw.map(SOURCE_MAP)
    p = p[p.source_group.notna()].reset_index(drop=True)
    p["z_benign"], p["z_attack"] = p.logit_neg, p.logit_pos
    p["margin"] = p.z_attack - p.z_benign
    return p[["sample_id", "source_group", "source_group_raw", "source_subgroup",
              "duplicate_group_id", "gt_attack", "z_benign", "z_attack", "p_unsafe",
              "margin", "token_length", "text"]].rename(columns={"p_unsafe": "p_attack"})


def _h(seed, tag, g):
    return int.from_bytes(hashlib.blake2b(f"{seed}|{tag}|{g}".encode(), digest_size=8).digest(),
                          "big") / 2 ** 64


def assign_split(df, seed=0):
    """(source_group, gt_attack) 층 안에서 duplicate_group 단위로 배정."""
    names = list(SPLIT_FRACS)
    out = {}
    for (src, y), sub in df.groupby(["source_group", "gt_attack"]):
        gs = sorted(sub.duplicate_group_id.unique(), key=lambda g: _h(seed, f"{src}|{y}", g))
        n = len(gs)
        edges, acc = [], 0.0
        for k in names[:-1]:
            acc += SPLIT_FRACS[k]; edges.append(int(round(n * acc)))
        bounds = [0] + edges + [n]
        for j, k in enumerate(names):
            for g in gs[bounds[j]:bounds[j + 1]]:
                out.setdefault(g, k)
    s = df.duplicate_group_id.map(out)
    assert s.notna().all()
    return s


def build(seed=0, save=True):
    p = load_population()
    p["split"] = assign_split(p, seed)
    bad = p.groupby("duplicate_group_id").split.nunique()
    assert (bad == 1).all(), f"중복 그룹 {int((bad>1).sum())}개가 분할에 걸침"
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        p.drop(columns=["text"]).to_parquet(DATA / "base_score_table.parquet", index=False)
        p[["sample_id", "source_group", "duplicate_group_id", "gt_attack", "split"]].to_parquet(
            DATA / "split_manifest.parquet", index=False)
    return p

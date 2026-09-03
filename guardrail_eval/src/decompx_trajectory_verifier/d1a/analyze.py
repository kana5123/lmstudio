"""PHASE D1A 집계: 자연 분포 / 길이 매칭 평가, paired bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.metrics import (core_metrics, paired_group_bootstrap,
                                                     recall_at_threshold, threshold_at_tp_loss)

PRED = ART / "phase_d1a/predictions"
MATCH = ART / "phase_d1a/length_matched_manifests"
OUT = RES / "phase_d1a"
PROTOCOLS = ["seen_source", "loso_wj", "loso_ps", "loso_qs"]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]
SEEDS = [0, 1, 2, 3, 4]


def load(proto, v, seed, split="test"):
    p = PRED / f"{proto}_{v}_seed{seed}_{split}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def matched_ids(proto, seed):
    p = MATCH / f"{proto}_seed{seed}_matched.parquet"
    return set(pd.read_parquet(p).sample_id) if p.exists() else set()


def results(matched=False):
    rows = []
    for proto in PROTOCOLS:
        for v in VARIANTS:
            for sd in SEEDS:
                te = load(proto, v, sd)
                if te is None:
                    continue
                if matched:
                    te = te[te.sample_id.isin(matched_ids(proto, sd))]
                    if not len(te):
                        continue
                m = core_metrics(te.y_fp.to_numpy(), te.p_fp.to_numpy())
                rows.append(dict(protocol=proto, variant=v, seed=sd, scope="pooled",
                                 source="ALL", **m))
                for s, g in te.groupby("source_group"):
                    ms = core_metrics(g.y_fp.to_numpy(), g.p_fp.to_numpy())
                    rows.append(dict(protocol=proto, variant=v, seed=sd, scope="source",
                                     source=s, **ms))
    return pd.DataFrame(rows)


def low_tp_loss(rates=(0.01, 0.05)):
    rows = []
    for proto in PROTOCOLS:
        for v in VARIANTS:
            for sd in SEEDS:
                va, te = load(proto, v, sd, "val"), load(proto, v, sd, "test")
                if va is None or te is None:
                    continue
                yv, pv = va.y_fp.to_numpy(), va.p_fp.to_numpy()
                yt, pt = te.y_fp.to_numpy(), te.p_fp.to_numpy()
                for r in rates:
                    thr = threshold_at_tp_loss(yv, pv, r)
                    fr, tl = recall_at_threshold(yt, pt, thr)
                    rows.append(dict(protocol=proto, variant=v, seed=sd, target_tp_loss=r,
                                     fp_recall=fr, actual_tp_loss=tl,
                                     n_test_tp=int((yt == 0).sum())))
    return pd.DataFrame(rows)


PAIRS = [("V2", "V0"), ("V2", "V1"), ("V3", "V2"), ("V4", "V3"), ("V1", "V0")]


def bootstrap(matched=False):
    rows = []
    for proto in PROTOCOLS:
        for a, b in PAIRS:
            for sd in SEEDS:
                A, B = load(proto, a, sd), load(proto, b, sd)
                if A is None or B is None:
                    continue
                if matched:
                    ids = matched_ids(proto, sd)
                    A, B = A[A.sample_id.isin(ids)], B[B.sample_id.isin(ids)]
                A = A.sort_values("sample_id").reset_index(drop=True)
                B = B.sort_values("sample_id").reset_index(drop=True)
                assert (A.sample_id.to_numpy() == B.sample_id.to_numpy()).all()
                y, g = A.y_fp.to_numpy(), A.duplicate_group_id.to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                for nm, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                    d = paired_group_bootstrap(y, A.p_fp.to_numpy(), B.p_fp.to_numpy(), g, fn,
                                               n=1000, seed=sd)
                    rows.append(dict(protocol=proto, comparison=f"{a}-{b}", metric=nm,
                                     seed=sd, **d))
    return pd.DataFrame(rows)

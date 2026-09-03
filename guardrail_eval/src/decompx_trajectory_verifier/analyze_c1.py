"""PHASE C1 집계 (§14-§17, §20).  예측 파일에서 모든 지표를 만든다."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.metrics import (core_metrics, group_bootstrap,
                                                     paired_group_bootstrap,
                                                     recall_at_threshold, threshold_at_tp_loss)

PRED = ART / "phase_c1/prediction_files"
OUT = RES / "phase_c1"
FLAGGED = None


def load(proto, model, seed, split):
    p = PRED / f"{proto}_{model}_seed{seed}_{split}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def flagged_ids():
    global FLAGGED
    if FLAGGED is None:
        q = pd.read_csv(RES / "b3_quarantine.csv")
        FLAGGED = set(q.sample_id) if len(q) else set()
    return FLAGGED


def per_split_metrics(df, boot=True, seed=0):
    y, p, g = df.y_fp.to_numpy(), df.p_fp.to_numpy(), df.duplicate_group_id.to_numpy()
    m = core_metrics(y, p)
    if boot and len(np.unique(y)) > 1:
        m["auroc_ci"] = group_bootstrap(y, p, g, roc_auc_score, seed=seed)
        m["auprc_ci"] = group_bootstrap(y, p, g, average_precision_score, seed=seed)
    return m


def collect(protocols, models, seeds):
    rows, per_src = [], []
    for proto in protocols:
        for mdl in models:
            for sd in seeds:
                te = load(proto, mdl, sd, "test")
                if te is None:
                    continue
                m = per_split_metrics(te, boot=False)
                rows.append(dict(protocol=proto, model=mdl, seed=sd, scope="pooled",
                                 source="ALL", **{k: v for k, v in m.items()}))
                for s, gdf in te.groupby("source_group"):
                    ms = core_metrics(gdf.y_fp.to_numpy(), gdf.p_fp.to_numpy())
                    per_src.append(dict(protocol=proto, model=mdl, seed=sd, source=s, **ms))
    return pd.DataFrame(rows), pd.DataFrame(per_src)


def low_tp_loss(protocols, models, seeds, rates=(0.01, 0.05, 0.005)):
    rows = []
    for proto in protocols:
        for mdl in models:
            for sd in seeds:
                va, te = load(proto, mdl, sd, "val"), load(proto, mdl, sd, "test")
                if va is None or te is None:
                    continue
                yv, pv = va.y_fp.to_numpy(), va.p_fp.to_numpy()
                yt, pt = te.y_fp.to_numpy(), te.p_fp.to_numpy()
                for r in rates:
                    thr = threshold_at_tp_loss(yv, pv, r)
                    fr, tl = recall_at_threshold(yt, pt, thr)
                    n_tp = int((yt == 0).sum())
                    rows.append(dict(protocol=proto, model=mdl, seed=sd, target_tp_loss=r,
                                     threshold=thr, fp_recall=fr, actual_tp_loss=tl,
                                     n_test_tp=n_tp,
                                     unstable=bool(n_tp * r < 10)))
    return pd.DataFrame(rows)


def paired(protocols, seeds, ref="A3", others=("A0", "M0")):
    rows = []
    for proto in protocols:
        for other in others:
            for sd in seeds:
                a, b = load(proto, ref, sd, "test"), load(proto, other, sd, "test")
                if a is None or b is None:
                    continue
                a = a.sort_values("sample_id").reset_index(drop=True)
                b = b.sort_values("sample_id").reset_index(drop=True)
                assert (a.sample_id.to_numpy() == b.sample_id.to_numpy()).all()
                y, g = a.y_fp.to_numpy(), a.duplicate_group_id.to_numpy()
                for nm, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                    d = paired_group_bootstrap(y, a.p_fp.to_numpy(), b.p_fp.to_numpy(), g, fn,
                                               seed=sd)
                    rows.append(dict(protocol=proto, comparison=f"{ref}-{other}", metric=nm,
                                     seed=sd, **d))
    return pd.DataFrame(rows)


def seed_stability(runs):
    g = runs.groupby(["protocol", "model"])
    out = g[["test_auroc", "test_auprc", "best_epoch", "best_val_macro_auprc"]].agg(
        ["mean", "std", "min", "max", "count"])
    return out


def audit_flag_sensitivity(protocols, models, seeds):
    fl = flagged_ids()
    rows = []
    for proto in protocols:
        for mdl in models:
            for sd in seeds:
                te = load(proto, mdl, sd, "test")
                if te is None:
                    continue
                full = core_metrics(te.y_fp.to_numpy(), te.p_fp.to_numpy())
                sub = te[~te.sample_id.isin(fl)]
                ex = core_metrics(sub.y_fp.to_numpy(), sub.p_fp.to_numpy())
                rows.append(dict(protocol=proto, model=mdl, seed=sd,
                                 n_flagged_in_test=int(te.sample_id.isin(fl).sum()),
                                 auroc_all=full["auroc"], auroc_excl=ex["auroc"],
                                 auprc_all=full["auprc"], auprc_excl=ex["auprc"],
                                 d_auroc=ex["auroc"] - full["auroc"],
                                 d_auprc=ex["auprc"] - full["auprc"]))
    return pd.DataFrame(rows)

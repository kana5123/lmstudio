"""PHASE E0 §5-§7: high-recall candidate gate 임계와 route rate.

tau_gate(rho) = 최소 source recall 이 rho 이상이 되는 가장 높은 threshold.
source 별 threshold 를 만들지 않는다 -- 추론 시점에 source 를 모른다고 가정한다.
임계는 gate_calib 에서만 고른다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.high_recall_attribution_cascade.config import GATE_RECALLS, RES


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tau_for_recall(calib, rho, score="p_attack"):
    """최소 source recall >= rho 를 만족하는 가장 높은 threshold.

    각 source 에서 recall >= rho 를 주는 최대 threshold 는 공격 점수의
    ceil(n_a * rho) 번째로 큰 값이다.  전체 제약은 그 최소값이다.
    """
    taus = []
    for s, g in calib.groupby("source_group"):
        a = np.sort(g.loc[g.gt_attack == 1, score].to_numpy())[::-1]
        if len(a) == 0:
            continue
        k = int(np.ceil(len(a) * rho))
        taus.append(a[min(k, len(a)) - 1])
    return float(min(taus))


def gate_report(df, tau, score="p_attack"):
    """주어진 threshold 에서 source 별/전체 지표."""
    rows = []
    for s, g in list(df.groupby("source_group")) + [("POOLED", df)]:
        cand = g[score] >= tau
        na = int((g.gt_attack == 1).sum()); nb = int((g.gt_attack == 0).sum())
        ta = int((cand & (g.gt_attack == 1)).sum()); fb = int((cand & (g.gt_attack == 0)).sum())
        rl, rh = wilson(ta, na); fl, fh = wilson(fb, nb)
        rows.append(dict(source_group=s, n=len(g), n_attack=na, n_benign=nb,
                         recall=ta / max(na, 1), fpr=fb / max(nb, 1),
                         recall_ci_lo=rl, recall_ci_hi=rh, fpr_ci_lo=fl, fpr_ci_hi=fh,
                         candidate_attack=ta, candidate_benign=fb,
                         n_candidates=ta + fb, route_rate=(ta + fb) / len(g),
                         min_fpr_step=1 / max(nb, 1)))
    return pd.DataFrame(rows)


def build(pop, recalls=GATE_RECALLS, extra_taus=(0.5,)):
    calib = pop[pop.split == "gate_calib"]
    taus = {f"rho={r}": tau_for_recall(calib, r) for r in recalls}
    taus |= {f"native={t}": float(t) for t in extra_taus}
    out = []
    for name, tau in taus.items():
        for split in ("gate_calib", "dev_test", "ALL"):
            d = pop if split == "ALL" else pop[pop.split == split]
            r = gate_report(d, tau)
            out.append(r.assign(gate=name, tau_gate=tau, eval_split=split))
    return taus, pd.concat(out, ignore_index=True)

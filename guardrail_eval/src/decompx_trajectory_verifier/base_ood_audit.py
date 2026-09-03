"""Base PromptGuard2 자체의 source 별 OOD 동작 감사.

★ 두 AUROC 를 절대 혼동하지 않는다.
  A. BASE DETECTOR AUROC
     모집단 = 해당 source 의 GT benign + GT attack 전부 (TP/FP/TN/FN 모두)
     양성 = GT attack,  점수 = z_attack - z_benign
  B. VERIFIER AUROC
     모집단 = base 가 ATTACK 이라 예측한 것만 (TP + FP)
     양성 = FP,  점수 = verifier P(FP)
PHASE C1 의 M0/A0/A3 는 전부 B 다.  M0 를 base detector AUROC 라고 부르지 않는다.

이번 분석에서 두 점수를 곱하거나 더해 combined AUROC 를 만들지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import RES

ROOT = Path(__file__).resolve().parents[2]
OUT = RES / "base_ood_audit"
SOURCES = ["wildjailbreak:adversarial", "promptshield:test", "piguard:Question Set"]
TARGETS = (0.001, 0.005, 0.01)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_population():
    c = pd.read_parquet(ROOT / "data/multisource_guard/canonical_samples.parquet",
                        columns=["sample_id", "source_group"])
    p = pd.read_parquet(ROOT / "data/decompx_verifier/pg2_predictions.parquet")
    p = p[(p.use == "MAIN") & p.length_ok].merge(c, on="sample_id", how="left")
    p = p[p.source_group.isin(SOURCES)].copy()
    p["margin"] = p["logit_pos"] - p["logit_neg"]    # z_attack - z_benign
    # 주의: 컬럼명 gt 는 pandas 의 gt() 메서드와 충돌한다 -> gt_attack 으로 바꾼다
    p = p.rename(columns={"gt": "gt_attack"})
    return p


def source_metrics(df):
    rows = []
    for s, g in df.groupby("source_group"):
        y, sc = g["gt_attack"].to_numpy(), g["margin"].to_numpy()
        cc = g.confusion_cell.value_counts()
        TP, FP, TN, FN = (int(cc.get(k, 0)) for k in ("TP", "FP", "TN", "FN"))
        na, nb = TP + FN, TN + FP
        rec, fpr = TP / max(na, 1), FP / max(nb, 1)
        prec = TP / max(TP + FP, 1)
        rows.append(dict(
            source_group=s, num_benign=nb, num_attack=na, n=len(g),
            TP=TP, FP=FP, TN=TN, FN=FN,
            base_auroc=roc_auc_score(y, sc), base_auprc=average_precision_score(y, sc),
            native_recall=rec, native_fpr=fpr, native_precision=prec,
            native_specificity=TN / max(nb, 1),
            native_recall_ci_lo=wilson(TP, na)[0], native_recall_ci_hi=wilson(TP, na)[1],
            native_fpr_ci_lo=wilson(FP, nb)[0], native_fpr_ci_hi=wilson(FP, nb)[1]))
    return pd.DataFrame(rows)


def low_fpr(df, targets=TARGETS):
    """FPR <= target 을 만족하는 threshold 중 Recall 최대인 지점.

    동률 처리: sklearn roc_curve 는 같은 점수를 가진 표본을 하나의 threshold 로 묶는다.
    따라서 선택된 threshold 에서 동점 표본은 모두 함께 양성으로 들어간다(부분 포함 없음).
    """
    rows = []
    for s, g in df.groupby("source_group"):
        y, sc = g["gt_attack"].to_numpy(), g["margin"].to_numpy()
        nb, na = int((y == 0).sum()), int((y == 1).sum())
        f, t, thr = roc_curve(y, sc)
        step = 1.0 / nb
        for tg in targets:
            ok = f <= tg
            if not ok.any():
                rows.append(dict(source_group=s, target_fpr=tg, feasible=False,
                                 min_fpr_step=step, benign_denominator=nb,
                                 attack_denominator=na,
                                 statistically_low_resolution=bool(step > tg)))
                continue
            i = int(np.argmax(np.where(ok, t, -1)))     # 제약 만족 중 recall 최대
            nfp, ntp = int(round(f[i] * nb)), int(round(t[i] * na))
            rl, rh = wilson(ntp, na); fl, fh = wilson(nfp, nb)
            rows.append(dict(
                source_group=s, target_fpr=tg, feasible=True, threshold=float(thr[i]),
                achieved_fpr=float(f[i]), achieved_recall=float(t[i]),
                fp_count=nfp, tp_count=ntp, benign_denominator=nb, attack_denominator=na,
                min_fpr_step=step, statistically_low_resolution=bool(step > tg),
                recall_ci_lo=rl, recall_ci_hi=rh, fpr_ci_lo=fl, fpr_ci_hi=fh))
    return pd.DataFrame(rows)


def score_distribution(df):
    rows = []
    for (s, y), g in df.groupby(["source_group", "gt_attack"]):
        m = g["margin"]
        rows.append(dict(source_group=s, gt_class="attack" if y == 1 else "benign", n=len(g),
                         mean=m.mean(), median=m.median(), p05=m.quantile(.05),
                         p25=m.quantile(.25), p75=m.quantile(.75), p95=m.quantile(.95)))
    return pd.DataFrame(rows)


def curve_points(df, n_max=400):
    roc, pr = [], []
    for s, g in df.groupby("source_group"):
        y, sc = g["gt_attack"].to_numpy(), g["margin"].to_numpy()
        f, t, thr = roc_curve(y, sc)
        idx = np.unique(np.linspace(0, len(f) - 1, min(n_max, len(f))).astype(int))
        roc.append(pd.DataFrame(dict(source_group=s, fpr=f[idx], tpr=t[idx], threshold=thr[idx])))
        p, r, th = precision_recall_curve(y, sc)
        j = np.unique(np.linspace(0, len(p) - 1, min(n_max, len(p))).astype(int))
        pr.append(pd.DataFrame(dict(source_group=s, precision=p[j], recall=r[j])))
    return pd.concat(roc, ignore_index=True), pd.concat(pr, ignore_index=True)


def fpr_resolution(df, targets=TARGETS):
    rows = []
    for s, g in df.groupby("source_group"):
        nb = int((g["gt_attack"] == 0).sum())
        for tg in targets:
            rows.append(dict(source_group=s, benign_n=nb, min_fpr_step=1.0 / nb, target_fpr=tg,
                             fp_allowed_at_target=int(np.floor(nb * tg)),
                             statistically_low_resolution=bool(1.0 / nb > tg)))
    return pd.DataFrame(rows)

"""§14-§16 지표.  positive = FP, negative = TP."""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)


def core_metrics(y, p):
    """y: 1=FP, 0=TP.  p: P(FP)."""
    out = dict(n=len(y), n_fp=int(y.sum()), n_tp=int((1 - y).sum()),
               fp_prevalence=float(y.mean()))
    if len(np.unique(y)) < 2:
        return out | dict(auroc=np.nan, auprc=np.nan, balanced_acc=np.nan,
                          brier=np.nan, logloss=np.nan)
    return out | dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                      balanced_acc=balanced_accuracy_score(y, (p >= 0.5).astype(int)),
                      brier=brier_score_loss(y, p), logloss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def threshold_at_tp_loss(y_val, p_val, rate):
    """검증셋에서 'TP 를 FP 로 잘못 판정하는 비율'이 rate 이하가 되는 최소 임계값."""
    tp_scores = np.sort(p_val[y_val == 0])[::-1]
    if len(tp_scores) == 0:
        return np.inf
    k = int(np.floor(len(tp_scores) * rate))
    return float(tp_scores[k]) if k < len(tp_scores) else float(tp_scores[-1] - 1e-9)


def recall_at_threshold(y, p, thr):
    m = p >= thr
    fp_recall = float((m & (y == 1)).sum() / max((y == 1).sum(), 1))
    tp_loss = float((m & (y == 0)).sum() / max((y == 0).sum(), 1))
    return fp_recall, tp_loss


def group_bootstrap(y, p, groups, fn, n=2000, seed=0, alpha=0.05):
    """duplicate_group_id 단위 부트스트랩 CI."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in uq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([gi[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    v = np.array(vals)
    return float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))


def paired_group_bootstrap(y, pa, pb, groups, fn, n=2000, seed=0, alpha=0.05):
    """같은 test 표본에 대한 두 모델 예측의 차이 CI (A - B)."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in uq}
    d = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([gi[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    d = np.array(d)
    return dict(delta=float(fn(y, pa) - fn(y, pb)),
                lo=float(np.nanpercentile(d, 100 * alpha / 2)),
                hi=float(np.nanpercentile(d, 100 * (1 - alpha / 2))),
                p_gt0=float((d > 0).mean()))

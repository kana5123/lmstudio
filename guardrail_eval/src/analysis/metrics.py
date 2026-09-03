"""공용 지표.  임계값·스케일러·방향은 전부 **train 에서만** 적합한다는 규약을 강제한다."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def auprc(y, s):
    return float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")


def pick_threshold(scores, labels, target_fpr):
    """rfpr.py:pick_threshold 와 **동일한 규칙**.  기존 벤치마크와 비교하려면 같아야 한다."""
    neg = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    if not neg:
        return 1.0
    k = int(len(neg) * target_fpr)
    return float(neg[k]) + 1e-12 if k < len(neg) else 0.0


def recall_fpr(scores, labels, thr):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    rec = sum(s >= thr for s in pos) / len(pos) if pos else float("nan")
    fpr = sum(s >= thr for s in neg) / len(neg) if neg else float("nan")
    return rec, fpr


def linear_probe(Xtr, ytr, Xte, yte, seed=0, C=1.0):
    """train 에서만 스케일러+로지스틱 적합, test 에서 평가.  누수 방지의 핵심 지점."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, random_state=seed, C=C).fit(sc.transform(Xtr), ytr)
    s = lr.predict_proba(sc.transform(Xte))[:, 1]
    return {"auroc": auroc(yte, s), "auprc": auprc(yte, s),
            "acc": float((s >= .5).astype(int).__eq__(yte).mean()),
            "f1": float(f1_score(yte, (s >= .5).astype(int)))}, s, (sc, lr)

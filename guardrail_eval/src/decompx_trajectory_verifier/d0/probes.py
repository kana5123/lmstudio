"""PHASE D0 §3-§4, §6, §10-§11.

source probe 는 반드시 TP-only / FP-only 로 나눠 수행한다.
그래야 source 분류가 TP/FP 라벨 차이를 지름길로 쓸 수 없다.
분할은 기존 seen_source_seed0 manifest 를 그대로 쓴다(그룹 단위, 새 분할 만들지 않음).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import ART, RES

D0 = ART / "phase_d0"
OUT = RES / "phase_d0"
SPLIT = ART / "phase_c1/split_manifests/seen_source_seed0.parquet"
SRC_ORDER = ["wildjailbreak:adversarial", "promptshield:test", "piguard:Question Set"]
SHORT = {"wildjailbreak:adversarial": "WJ", "promptshield:test": "PS", "piguard:Question Set": "QS"}


def load():
    meta = pd.read_parquet(D0 / "meta.parquet")
    sp = pd.read_parquet(SPLIT)[["sample_id", "split"]].rename(columns={"split": "c1_split"})
    meta = meta.merge(sp, on="sample_id", how="left")
    return meta, np.load(D0 / "h_layers.npy"), np.load(D0 / "tok_norm_stats.npy")


def source_probe(X, meta, cell, seed=0):
    """cell='TP' 또는 'FP' 로 모집단을 고정한 뒤 source 3분류.  chance macro accuracy ~ 1/3."""
    m = (meta.confusion_cell == cell).to_numpy()
    tr = m & (meta.c1_split == "train").to_numpy()
    te = m & (meta.c1_split == "test").to_numpy()
    y = meta.source_group.to_numpy()
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=8,
                             random_state=seed).fit(sc.transform(X[tr]), y[tr])
    p = clf.predict(sc.transform(X[te]))
    pr = clf.predict_proba(sc.transform(X[te]))
    ovr = {}
    for i, c in enumerate(clf.classes_):
        yy = (y[te] == c).astype(int)
        ovr[f"auroc_{SHORT[c]}"] = roc_auc_score(yy, pr[:, i]) if len(np.unique(yy)) > 1 else np.nan
    return dict(cell=cell, n_train=int(tr.sum()), n_test=int(te.sum()),
                macro_f1=f1_score(y[te], p, average="macro"),
                balanced_acc=balanced_accuracy_score(y[te], p), **ovr)


def layerwise_source_probe(H, meta):
    rows = []
    for l in range(H.shape[1]):
        for cell in ("TP", "FP"):
            r = source_probe(H[:, l], meta, cell)
            if r:
                rows.append(dict(layer=l + 1, **r))
    return pd.DataFrame(rows)


def fp_tp_direction(X, meta, name):
    """d_s = mean_FP(source=s) - mean_TP(source=s), source 쌍별 코사인."""
    d = {}
    for s in SRC_ORDER:
        m = (meta.source_group == s).to_numpy()
        fp = X[m & (meta.confusion_cell == "FP").to_numpy()].mean(0)
        tp = X[m & (meta.confusion_cell == "TP").to_numpy()].mean(0)
        d[s] = fp - tp
    out = dict(representation=name)
    for i, a in enumerate(SRC_ORDER):
        for b in SRC_ORDER[i + 1:]:
            u, v = d[a], d[b]
            out[f"cos_{SHORT[a]}_{SHORT[b]}"] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    for s in SRC_ORDER:
        out[f"norm_{SHORT[s]}"] = float(np.linalg.norm(d[s]))
    return out


def c_norm_audit(meta, S):
    """§10 A: ||C_lk|| 표본별 통계, B: ||sum_k C_lk||."""
    H = np.load(D0 / "h_layers.npy")
    hn = np.linalg.norm(H, axis=-1)                    # [n,L]
    rows = []
    for s in SRC_ORDER:
        for cell in ("TP", "FP"):
            m = ((meta.source_group == s) & (meta.confusion_cell == cell)).to_numpy()
            for l in range(S.shape[1]):
                tk = S[m, l]                            # [n_s, 5] median,p25,p75,p95,mean
                rows.append(dict(source_group=s, confusion_cell=cell, layer=l + 1, n=int(m.sum()),
                                 tok_norm_median=float(np.median(tk[:, 0])),
                                 tok_norm_p25=float(np.median(tk[:, 1])),
                                 tok_norm_p75=float(np.median(tk[:, 2])),
                                 tok_norm_p95=float(np.median(tk[:, 3])),
                                 tok_norm_mean=float(tk[:, 4].mean()),
                                 cls_norm_median=float(np.median(hn[m, l])),
                                 cls_norm_p25=float(np.percentile(hn[m, l], 25)),
                                 cls_norm_p75=float(np.percentile(hn[m, l], 75)),
                                 cls_norm_p95=float(np.percentile(hn[m, l], 95)),
                                 cls_norm_mean=float(hn[m, l].mean())))
    df = pd.DataFrame(rows)
    # source 쌍별 standardized mean difference (Cohen's d, TP/FP 합쳐서)
    smd = []
    for l in range(S.shape[1]):
        for i, a in enumerate(SRC_ORDER):
            for b in SRC_ORDER[i + 1:]:
                x = hn[(meta.source_group == a).to_numpy(), l]
                y = hn[(meta.source_group == b).to_numpy(), l]
                sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) + 1e-12
                smd.append(dict(layer=l + 1, pair=f"{SHORT[a]}-{SHORT[b]}",
                                smd_cls_norm=float((x.mean() - y.mean()) / sp)))
    return df, pd.DataFrame(smd)


def length_control(meta):
    """§11 대조군: 토큰 길이만으로 source / TP-FP 를 얼마나 맞히나."""
    tr = (meta.c1_split == "train").to_numpy(); te = (meta.c1_split == "test").to_numpy()
    X = meta.token_length.to_numpy().reshape(-1, 1).astype(np.float32)
    rows = []
    for cell in ("TP", "FP"):
        m = (meta.confusion_cell == cell).to_numpy()
        y = meta.source_group.to_numpy()
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr & m], y[tr & m])
        p = c.predict(X[te & m])
        rows.append(dict(task=f"source (cell={cell})", metric="macro_f1",
                         value=f1_score(y[te & m], p, average="macro"),
                         balanced_acc=balanced_accuracy_score(y[te & m], p)))
    for s in ["ALL"] + SRC_ORDER:
        m = np.ones(len(meta), bool) if s == "ALL" else (meta.source_group == s).to_numpy()
        y = meta.y_fp.to_numpy()
        if len(np.unique(y[tr & m])) < 2:
            continue
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr & m], y[tr & m])
        rows.append(dict(task=f"FP-vs-TP ({SHORT.get(s, s)})", metric="auroc",
                         value=roc_auc_score(y[te & m], c.predict_proba(X[te & m])[:, 1]),
                         balanced_acc=np.nan))
    return pd.DataFrame(rows)

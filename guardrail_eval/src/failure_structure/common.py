"""failure_structure 공용 모듈.

연구 질문(바뀐 것):
  "PromptGuard2 은닉표현에 base prediction 이 맞을지 틀릴지를 예측하는
   failure-associated 정보가 존재하는가?"
  단일 방향/중심차/선형 평균이동 형태라고 **가정하지 않는다.**

데이터셋 등급 (실측 cell 수 기준):
  MAIN   wildjailbreak      (최소 cell 3393)
  MAIN   promptshield_test  (최소 cell  455)
  WEAK   questionset        (최소 cell  105)  — 결과에 항상 불안정 표시
  CONFOUNDED_DIAGNOSTIC jailbreaksovertime (최소 cell 119, 오탐 100% wildchat 확정)
  EXCLUDE promptshield_train (FP 22, 게다가 promptshield_test 와 같은 데이터셋)
"""
import hashlib, itertools, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import glob
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/direction_debug"
RES = ROOT / "results/failure_structure"
PLOT = ROOT / "plots/failure_structure"
OUTA = ROOT / "artifacts/failure_structure"

CELLS = ("TP", "FP", "TN", "FN")
CORRECT = ("TP", "TN")          # y_error = 0
INCORRECT = ("FP", "FN")        # y_error = 1
ATTACK = ("TP", "FN")
PRED_UNSAFE = ("TP", "FP")

MAIN = ["wildjailbreak", "promptshield_test"]
WEAK = ["questionset"]
CONFOUNDED = ["jailbreaksovertime"]
EXCLUDED = ["promptshield_train"]
USABLE = MAIN + WEAK                      # cross-dataset / LODO 후보
ALL_ANALYSED = MAIN + WEAK + CONFOUNDED

SEEDS = [0, 1, 2]
TEST_FRAC = 0.30
INNER_VAL_FRAC = 0.25                     # TRAIN 안에서 hyperparameter 선택용
EPS = 1e-12


# ------------------------------------------------------------------ 적재
def load():
    fs = sorted(glob.glob(str(ART / "cellhidden_*of*.pt")))
    assert fs, "cellhidden 없음"
    ds = [torch.load(f, weights_only=False) for f in fs]
    out = {"h": torch.cat([d["h"] for d in ds]).numpy().astype(np.float32)}
    for k in ("dataset", "cell", "dup", "sample_id", "split_role"):
        out[k] = np.array([x for d in ds for x in d[k]])
    for k in ("logit_unsafe", "logit_benign", "text_len"):
        out[k] = torch.cat([d[k] for d in ds]).numpy().astype(np.float64)
    return out


def representations(h):
    """지시문 6절 R1~R6.  h:(n, L+1, D)  ->  {이름: (n, n_layer, dim)}"""
    g = h[:, 1:] - h[:, :-1]
    def l2n(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return {
        "R1_h":        (h,                                     [f"h_L{i}" for i in range(h.shape[1])]),
        "R2_g":        (g,                                     [f"g_L{i}->L{i+1}" for i in range(g.shape[1])]),
        "R3_h_norm":   (l2n(h),                                [f"h_L{i}" for i in range(h.shape[1])]),
        "R4_g_norm":   (l2n(g),                                [f"g_L{i}->L{i+1}" for i in range(g.shape[1])]),
        "R5_cat_hh":   (np.concatenate([h[:, :-1], h[:, 1:]], -1),
                        [f"cat_h{i}_h{i+1}" for i in range(h.shape[1] - 1)]),
        "R6_cat_hg":   (np.concatenate([h[:, :-1], g], -1),
                        [f"cat_h{i}_g{i+1}" for i in range(h.shape[1] - 1)]),
    }


# ------------------------------------------------------------------ 분할
def group_split(dup, cell, seed, frac=TEST_FRAC, salt="fs"):
    """중복그룹(dup)을 쪼개지 않으면서 cell 비율 보존.  같은 dup 은 항상 같은 쪽."""
    rng = np.random.default_rng(seed)
    first = {}
    for i, d in enumerate(dup):
        if d not in first:
            first[d] = cell[i]
    groups = np.array(list(first.keys()))
    strat = np.array([first[g] for g in groups])
    hold = set()
    for c in CELLS:
        gc = groups[strat == c]
        if len(gc) == 0:
            continue
        k = int(round(len(gc) * frac))
        hold |= set(rng.permutation(gc)[:k].tolist())
    is_hold = np.array([d in hold for d in dup])
    return ~is_hold, is_hold


def cell_weights(cell):
    """지시문 8절: 네 cell 이 **동일 총가중**을 갖게 한다.
    TN 이 많다고 correctness probe 를 지배하지 않도록."""
    w = np.zeros(len(cell), dtype=np.float64)
    for c in CELLS:
        m = cell == c
        if m.sum():
            w[m] = 1.0 / m.sum()
    return w * (len(cell) / w.sum())        # 평균 가중 1 로 스케일


def y_error(cell):
    """1 = incorrect (FP,FN),  0 = correct (TP,TN)."""
    return np.isin(cell, INCORRECT).astype(int)


# ------------------------------------------------------------------ 지표
def auroc(y, s):
    """**부호를 뒤집지 않는다.**  0.5 미만도 그대로 보고."""
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def auprc(y, s):
    return float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan"), float("nan")
    o = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        o.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def boot_delta_ci(y, s1, s2, n=1000, seed=0):
    """두 점수의 AUROC 차이에 대한 부트스트랩 구간 (짝지어 재표본)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan"), float("nan")
    o = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        o.append(roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def fdr_bh(pvals, q=0.05):
    """Benjamini-Hochberg.  반환: (기각여부 배열, 조정 p)"""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out_rej = np.zeros(len(p), bool); out_adj = np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out_rej, out_adj
    pp = p[idx]; order = np.argsort(pp); ps = pp[order]; m = len(ps)
    adj = ps * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_adj[idx[order]] = adj
    out_rej[idx[order]] = adj <= q
    return out_rej, out_adj


def wcsv(path, rows):
    if not rows:
        print(f"  [빈 결과] {path}")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  저장 {path.name} ({len(rows)}행)")

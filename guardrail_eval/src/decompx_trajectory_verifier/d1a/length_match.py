"""§9-§10 길이 매칭 test 집합 생성과 감사.

각 (source, split, 길이구간) 에서 n_match = min(num_TP, num_FP) 만큼 TP/FP 를 같은 수로 뽑는다.
선택은 hash(sample_id, seed) 순서로 결정론적이며, 한 matched set 안에서 표본을 중복 사용하지 않는다.
test 결과를 보고 매칭을 조정하지 않는다.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import ART, DATA, RES

BINS = [(1 + 32 * i, 32 * (i + 1)) for i in range(16)]      # 1-32 ... 481-512
MAN = ART / "phase_c1/split_manifests"
OUTMAN = ART / "phase_d1a/length_matched_manifests"
PROTOCOLS = ["seen_source", "loso_wj", "loso_ps", "loso_qs"]


def _key(sid, seed):
    return hashlib.blake2b(f"{seed}|{sid}".encode(), digest_size=8).hexdigest()


def bin_of(n):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= n <= hi:
            return i
    return len(BINS) - 1


def build(protocol, seed):
    man = pd.read_parquet(MAN / f"{protocol}_seed{seed}.parquet")
    core = pd.read_parquet(DATA / "core_tp_fp.parquet")[["sample_id", "token_length"]]
    te = man[man.split == "test"].merge(core, on="sample_id", how="left")
    te["bin"] = te.token_length.map(bin_of)
    te["k"] = [_key(s, seed) for s in te.sample_id]
    keep = []
    for (src, b), g in te.groupby(["source_group", "bin"]):
        tp = g[g.confusion_cell == "TP"].sort_values("k")
        fp = g[g.confusion_cell == "FP"].sort_values("k")
        n = min(len(tp), len(fp))
        if n == 0:
            continue
        keep.append(pd.concat([tp.head(n), fp.head(n)]))
    if not keep:
        return None
    out = pd.concat(keep, ignore_index=True)
    assert out.sample_id.is_unique, "matched set 에 중복 표본이 있다"
    out["split"] = "test"
    return out[["sample_id", "duplicate_group_id", "source_group", "source_subgroup",
                "confusion_cell", "split", "token_length", "bin"]]


def audit(matched):
    rows = []
    for (src, b), g in matched.groupby(["source_group", "bin"]):
        c = g.confusion_cell.value_counts()
        rows.append(dict(source_group=src, bin=f"{BINS[b][0]}-{BINS[b][1]}",
                         TP=int(c.get("TP", 0)), FP=int(c.get("FP", 0)),
                         balanced=bool(c.get("TP", 0) == c.get("FP", 0))))
    stat = []
    for (src, cell), g in matched.groupby(["source_group", "confusion_cell"]):
        t = g.token_length
        stat.append(dict(source_group=src, confusion_cell=cell, n=len(g), mean=t.mean(),
                         median=t.median(), p25=t.quantile(.25), p75=t.quantile(.75),
                         p95=t.quantile(.95)))
    return pd.DataFrame(rows), pd.DataFrame(stat)


def length_only_control(matched):
    """§10: matched 집합에서 길이만으로 FP-vs-TP AUROC.  0.55 초과면 잔여 신호 표시."""
    rows = []
    for src, g in matched.groupby("source_group"):
        y = (g.confusion_cell == "FP").astype(int).to_numpy()
        x = g.token_length.to_numpy().reshape(-1, 1).astype(np.float32)
        if len(np.unique(y)) < 2:
            continue
        c = LogisticRegression(max_iter=1000).fit(x, y)
        au = roc_auc_score(y, c.predict_proba(x)[:, 1])
        rows.append(dict(source_group=src, n=len(g), length_only_auroc=au,
                         residual_length_signal=bool(au > 0.55)))
    y = (matched.confusion_cell == "FP").astype(int).to_numpy()
    x = matched.token_length.to_numpy().reshape(-1, 1).astype(np.float32)
    c = LogisticRegression(max_iter=1000).fit(x, y)
    au = roc_auc_score(y, c.predict_proba(x)[:, 1])
    rows.append(dict(source_group="POOLED", n=len(matched), length_only_auroc=au,
                     residual_length_signal=bool(au > 0.55)))
    return pd.DataFrame(rows)


def main(seeds=(0, 1, 2, 3, 4)):
    OUTMAN.mkdir(parents=True, exist_ok=True)
    (RES / "phase_d1a").mkdir(parents=True, exist_ok=True)
    A, S, L = [], [], []
    for proto in PROTOCOLS:
        for seed in seeds:
            m = build(proto, seed)
            if m is None:
                continue
            m.to_parquet(OUTMAN / f"{proto}_seed{seed}_matched.parquet", index=False)
            a, s = audit(m)
            A.append(a.assign(protocol=proto, seed=seed))
            S.append(s.assign(protocol=proto, seed=seed))
            L.append(length_only_control(m).assign(protocol=proto, seed=seed))
    pd.concat(A, ignore_index=True).to_csv(RES / "phase_d1a/length_matching_audit.csv", index=False)
    pd.concat(S, ignore_index=True).to_csv(RES / "phase_d1a/length_matched_stats.csv", index=False)
    pd.concat(L, ignore_index=True).to_csv(RES / "phase_d1a/length_only_matched_control.csv",
                                           index=False)
    return pd.concat(A, ignore_index=True), pd.concat(S, ignore_index=True), pd.concat(L, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    A, S, L = main()
    print(f"길이 구간 균형: 전체 {len(A)}칸 중 불균형 {int((~A.balanced).sum())}칸")
    print("\n=== matched 집합 크기 (seed 0) ===")
    a0 = A[A.seed == 0]
    print(a0.groupby(["protocol", "source_group"])[["TP", "FP"]].sum().to_string())
    print("\n=== matched 길이 통계 (seed 0) ===")
    print(S[S.seed == 0].round(1).to_string(index=False))
    print("\n=== 길이 단독 대조군 (matched, 기준 0.55) ===")
    print(L.groupby(["protocol", "source_group"]).agg(
        auroc_mean=("length_only_auroc", "mean"), auroc_max=("length_only_auroc", "max"),
        residual=("residual_length_signal", "any")).round(4).to_string())

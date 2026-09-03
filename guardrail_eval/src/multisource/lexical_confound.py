"""어휘 교란 진단 (지시문 14절 보강).

같은 source_group 안에서 TP 와 FP 가 **원문 어휘만으로** 얼마나 갈리는지 잰다.
JailbreaksOverTime 에서 표현 방향이 사실은 말뭉치 방향이었던 전례가 있으므로,
"이 출처의 TP/FP 가 문체/어휘만으로 이미 거의 완벽히 갈리는가"를 미리 확인한다.

이건 표현/방향 학습이 아니라 **표면 통계**다. 은닉표현·DecompX·방향을 쓰지 않는다.
어휘로 이미 AUROC≈1.0 이면, 나중에 표현 방향이 잘 갈려도 그것이
'내부 이동 구조'인지 '문체'인지 구별할 수 없다는 경고다.
"""
import sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / "data/multisource_guard", ROOT / "results/multisource_guard"
MIN_N = 20


def main():
    cc = pd.read_parquet(DATA / "confusion_cells.parquet")
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[["sample_id", "text"]]
    df = cc.merge(can, on="sample_id", how="left")
    df = df[df["confusion_cell"].isin(["TP", "FP"])]
    rows = []
    print("=== 같은 출처 안 TP vs FP 를 '원문 어휘만'으로 가를 수 있는가 (5겹 교차검증) ===")
    print(f"{'source_group':44} {'TP':>6} {'FP':>6} {'어휘 AUROC':>11}  판정")
    for g, sub in df.groupby("source_group"):
        y = (sub["confusion_cell"] == "TP").astype(int).values
        if y.sum() < MIN_N or (1 - y).sum() < MIN_N:
            continue
        X = sub["text"].astype(str).values
        aus = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                max_features=100000)
            A = v.fit_transform(X[tr])
            lr = LogisticRegression(max_iter=2000, random_state=0).fit(A, y[tr])
            aus.append(roc_auc_score(y[te], lr.predict_proba(v.transform(X[te]))[:, 1]))
        au = float(np.mean(aus))
        verdict = ("어휘로 거의 완전 분리 — 표현 신호와 구별 어려움" if au > 0.98 else
                   "어휘 분리 강함 — 주의" if au > 0.90 else
                   "어휘만으로는 부분 분리 — 표현 신호 검증 여지 있음")
        rows.append({"source_group": g, "n_TP": int(y.sum()), "n_FP": int((1 - y).sum()),
                     "lexical_auroc_cv5": au, "lexical_auroc_std": float(np.std(aus)),
                     "interpretation": verdict})
        print(f"{g:44} {int(y.sum()):6} {int((1-y).sum()):6} {au:11.4f}  {verdict}")
    pd.DataFrame(rows).to_csv(RES / "lexical_confound_probe.csv", index=False)
    print(f"\n저장 -> {RES/'lexical_confound_probe.csv'}")


if __name__ == "__main__":
    main()

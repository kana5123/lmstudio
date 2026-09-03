"""오탐(FP) 셀의 라벨 품질 점검.

FP 는 "GT 정상인데 PG2 가 UNSAFE 라 한 것"이다.  그런데 원본 데이터셋이 실제로는
탈옥문을 정상(label 0)으로 잘못 달아 놓았다면, 그 FP 는 **모델의 오탐이 아니라
데이터의 라벨 잡음**이다.  그런 셀로 TP-vs-FP 방향을 학습하면 방향이 오염된다.

여기서는 사전 정의한 탈옥/주입 표지 문구가 FP 텍스트에 실제로 들어 있는지 센다.
표지는 보수적으로(명백한 것만) 고른다.
"""
import re, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / "data/multisource_guard", ROOT / "results/multisource_guard"

MARKERS = {
    "ignore_previous": r"(ignore|forget|disregard)\s+(and\s+forget\s+)?(all\s+)?(the\s+)?(your\s+)?"
                       r"(previous|prior|above|earlier|preceding)\s+(instruction|prompt|rule|direction)",
    "dan_persona": r"\b(DAN|DOGA|AIM|STAN|DUDE|JailBreak|Developer Mode)\b",
    "do_anything_now": r"do anything now",
    "no_restrictions": r"(no (ethical )?(restrictions|filters|guidelines|limitations)|"
                       r"broken free|without any (restrictions|filters))",
    "new_instructions": r"(new instructions?|follow these instructions? instead)",
}


def main():
    cc = pd.read_parquet(DATA / "confusion_cells.parquet")
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[["sample_id", "text"]]
    df = cc.merge(can, on="sample_id", how="left")
    rows = []
    print("=== 셀별 '명백한 탈옥/주입 표지' 포함 비율 ===")
    print(f"{'source_group':44} {'셀':>3} {'n':>6} {'표지포함':>8}  주요 표지")
    for g, sub in df.groupby("source_group"):
        if not {"TP", "FP"} <= set(sub["confusion_cell"].unique()):
            continue
        if (sub["confusion_cell"] == "FP").sum() < 20:
            continue
        for cell in ("TP", "FP", "TN"):
            s = sub[sub["confusion_cell"] == cell]
            if len(s) == 0:
                continue
            hit = {}
            any_hit = pd.Series(False, index=s.index)
            for nm, pat in MARKERS.items():
                m = s["text"].astype(str).str.contains(pat, case=False, regex=True, na=False)
                hit[nm] = int(m.sum()); any_hit |= m
            rate = float(any_hit.mean())
            rows.append({"source_group": g, "cell": cell, "n": len(s),
                         "marker_rate": rate, **{f"n_{k}": v for k, v in hit.items()}})
            top = sorted(hit.items(), key=lambda x: -x[1])[:3]
            print(f"{g:44} {cell:>3} {len(s):6} {rate*100:7.1f}%  "
                  + ", ".join(f"{k}={v}" for k, v in top if v))
    pd.DataFrame(rows).to_csv(RES / "fp_label_quality_probe.csv", index=False)
    print(f"\n저장 -> {RES/'fp_label_quality_probe.csv'}")


if __name__ == "__main__":
    main()

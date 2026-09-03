"""NotInject 전체(one/two/three = 339건, 전부 정상)로 오탐율 + 지연시간 측정.

배치 없이 한 건씩 처리한다 -- 요청 하나가 실제로 겪는 지연을 재기 위해서.
공격 샘플이 0건이므로 이 데이터로는 FPR 만 측정 가능하다.
"""
import json, statistics as st, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd
from guards import GUARDS

SNAP = ("/home/kana5123/.cache/huggingface/hub/datasets--leolee99--NotInject/"
        "snapshots/847ae76cf8fea5ed325429e569ae8cfef022d2e0/data/")
SUBSETS = ["one", "two", "three"]
WARMUP = 3
OUT = Path(__file__).resolve().parent / "results"


def load_all():
    rows = []
    for s in SUBSETS:
        df = pd.read_parquet(f"{SNAP}NotInject_{s}-00000-of-00001.parquet")
        for i, r in df.iterrows():
            rows.append({"subset": s, "idx": int(i), "text": r["prompt"],
                         "category": r["category"], "triggers": list(r["word_list"])})
    assert len(rows) == 339, len(rows)
    return rows


def main():
    name = sys.argv[1]
    rows = load_all()
    guard = GUARDS[name]()
    for r in rows[:WARMUP]:
        guard.predict([r["text"]])

    for r in rows:
        t = time.perf_counter()
        (flag, raw), = guard.predict([r["text"]])
        r["ms"] = (time.perf_counter() - t) * 1000
        r["flag"], r["raw"] = flag, raw

    OUT.mkdir(exist_ok=True)
    p = OUT / f"notinject_{name}.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ms = sorted(r["ms"] for r in rows)
    fp = sum(r["flag"] for r in rows)
    print(f"\n== {name}")
    print(f"   FPR 전체   {fp}/339 = {fp/339:.4f}")
    for s in SUBSETS:
        sub = [r for r in rows if r["subset"] == s]
        print(f"   FPR {s:5}  {sum(r['flag'] for r in sub):2}/{len(sub)} = {sum(r['flag'] for r in sub)/len(sub):.4f}")
    print(f"   지연 평균 {st.mean(ms):.1f}ms  중앙값 {st.median(ms):.1f}ms  p95 {ms[int(len(ms)*.95)]:.1f}ms")
    print(f"   → {p}")


if __name__ == "__main__":
    main()

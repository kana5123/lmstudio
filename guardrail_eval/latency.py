"""요청 하나당 지연시간(배치 1). 정확도 측정과 분리해서 잰다.

표본: JailbreaksOverTime 50건 + PIArena 50건 (길이 분포가 다르므로 둘 다).
워밍업 3건은 버리고 중앙값 / p95 를 ms 로 보고.
"""
import json, statistics as st, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from benchmarks import jailbreaks_over_time, piarena_direct
from guards import GUARDS

N_PER_BENCH, WARMUP = 50, 3


def main():
    name = sys.argv[1]
    cases = jailbreaks_over_time(N_PER_BENCH) + piarena_direct()[: N_PER_BENCH * 2 : 2]
    guard = GUARDS[name]()
    for c in cases[:WARMUP]:
        guard.predict([c.text])
    rec = []
    for c in cases:
        t = time.perf_counter()
        guard.predict([c.text])
        rec.append({"bench": c.bench, "chars": len(c.text),
                    "ms": (time.perf_counter() - t) * 1000})
    ms = sorted(r["ms"] for r in rec)
    out = {"guard": name, "n": len(rec),
           "median_ms": round(st.median(ms), 1),
           "p95_ms": round(ms[int(len(ms) * 0.95)], 1),
           "mean_ms": round(st.mean(ms), 1),
           "per_bench": {b: round(st.median([r["ms"] for r in rec if r["bench"] == b]), 1)
                         for b in dict.fromkeys(r["bench"] for r in rec)}}
    p = Path("results") / f"latency_{name}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

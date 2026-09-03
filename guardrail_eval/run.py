"""가드레일 하나를 3개 벤치마크에 돌려 results/<이름>.jsonl 저장.

사용법:  CUDA_VISIBLE_DEVICES=7 python run.py qwen3guard
"""
import json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from benchmarks import load, VAL_SIZE
from guards import GUARDS

OUT = Path(__file__).resolve().parent / "results"
BATCH = 16


def main():
    name = sys.argv[1]
    n_jot = VAL_SIZE if "--full" in sys.argv else 100
    cases = load(n_jot)
    guard = GUARDS[name]()
    t0 = time.time()
    rows = []
    for i in range(0, len(cases), BATCH):
        chunk = cases[i : i + BATCH]
        for c, (flag, raw) in zip(chunk, guard.predict([c.text for c in chunk])):
            rows.append({"bench": c.bench, "label": c.label, "meta": c.meta,
                         "flag": flag, "raw": raw, "text": c.text})
        print(f"  {min(i+BATCH, len(cases))}/{len(cases)}  ({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{name}  {len(rows)}건  {time.time()-t0:.0f}s  → {p}")


if __name__ == "__main__":
    main()

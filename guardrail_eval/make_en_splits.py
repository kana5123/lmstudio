"""한국어 분할과 '같은 레코드·같은 분할'의 영어판을 만든다.

단일 변수 격리: 모델·하이퍼파라미터·분할·라벨 전부 동일, 입력 언어만 ko -> en.
"""
import json
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
SRC = HERE / "safeguard_ko_verified.json"
KO, EN = HERE / "data_ko", HERE / "data_en"
EN.mkdir(exist_ok=True)

recs = json.load(open(SRC, encoding="utf-8"))
ko2en = {}
for r in recs:
    ko2en.setdefault(r["text_ko"], r["text_en"])
print(f"원본 {len(recs)}건 -> 고유 한국어 {len(ko2en)}건")

seen = {}
for split in ("train", "val", "test"):
    rows = [json.loads(l) for l in open(KO / f"{split}.jsonl", encoding="utf-8")]
    out, miss = [], 0
    for r in rows:
        en = ko2en.get(r["text"])
        if en is None:
            miss += 1
            continue
        out.append({"text": en, "label": r["label"]})
    (EN / f"{split}.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    seen[split] = {o["text"] for o in out}
    pos = sum(o["label"] for o in out)
    print(f"{split:6} 한국어 {len(rows)} -> 영어 {len(out)}  (매칭 실패 {miss})"
          f"  정상 {len(out)-pos} / 공격 {pos}")

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    n = len(seen[a] & seen[b])
    print(f"누수 {a}∩{b} = {n}건")
    assert n == 0, f"{a}∩{b} 누수 {n}건"

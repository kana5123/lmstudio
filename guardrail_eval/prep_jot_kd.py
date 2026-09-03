"""증류용 영어 데이터: JailbreaksOverTime 2023-02~11 (2023-12 평가셋 제외).

라벨은 쓰지 않는다 -- 교사(Prompt Guard 2)가 채점한다.
시간 분할이므로 평가셋(12월)과 구조적으로 겹치지 않는다.
"""
import json, datetime as dt
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from rfpr import JOT

HERE = Path(__file__).resolve().parent
rows = json.load(open(JOT))
for r in rows:
    r["ym"] = dt.datetime.utcfromtimestamp(r["timestamp"]).strftime("%Y-%m")
train = [r for r in rows if r["ym"] < "2023-12"]
test  = [r for r in rows if r["ym"] == "2023-12"]

seen, dd = set(), []
for r in train:
    if r["prompt"] not in seen: seen.add(r["prompt"]); dd.append(r)
tset = {r["prompt"] for r in test}
before = len(dd)
dd = [r for r in dd if r["prompt"] not in tset]      # 12월과 겹치는 문장 제거
print(f"2023-02~11  {len(train)} → 중복제거 {before} → 12월 중복 제거 {len(dd)}")
print(f"  (참고 라벨) 공격 {sum(r['label'] for r in dd)} / 정상 {len(dd)-sum(r['label'] for r in dd)}")

o = HERE / "data_jot"; o.mkdir(exist_ok=True)
import random; random.Random(0).shuffle(dd)
n_val = 2000
for nm, part in (("val", dd[:n_val]), ("train", dd[n_val:])):
    (o/f"{nm}.jsonl").write_text(
        "".join(json.dumps({"text": r["prompt"], "label": r["label"]},
                           ensure_ascii=False)+"\n" for r in part), encoding="utf-8")
    print(f"  data_jot/{nm}.jsonl {len(part)}건")
# 시험셋(영어, 12월) — 증류 결과를 영어에서 먼저 확인하는 용도
seen2, td = set(), []
for r in test:
    if r["prompt"] not in seen2: seen2.add(r["prompt"]); td.append(r)
(o/"test.jsonl").write_text(
    "".join(json.dumps({"text": r["prompt"], "label": r["label"]},
                       ensure_ascii=False)+"\n" for r in td), encoding="utf-8")
print(f"  data_jot/test.jsonl {len(td)}건 (공격 {sum(r['label'] for r in td)})")
tr = {r["prompt"] for r in dd}
print(f"누수 학습 ∩ 시험 = {len(tr & {r['prompt'] for r in td})}건")

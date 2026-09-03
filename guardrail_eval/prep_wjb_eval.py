"""문체를 통제한 한국어 평가셋 재료 준비.

공격 = WildJailbreak adversarial_harmful   (제일브레이크 + 유해 의도)
정상 = WildJailbreak adversarial_benign    (제일브레이크풍이나 무해)
둘 다 같은 데이터셋·같은 생성 방식이므로 문체로 구분할 수 없다.
"""
import json, random, statistics as st
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
CAP, N = 4000, 400
d = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
buckets = {"adversarial_harmful": [], "adversarial_benign": []}
for r in d:
    t = r["data_type"]
    if t not in buckets: continue
    txt = (r["adversarial"] or "").strip()
    if txt and len(txt) <= CAP: buckets[t].append(txt)
for k, v in buckets.items():
    v[:] = list(dict.fromkeys(v))
    print(f"{k:22} {len(v)}건 (≤{CAP}자)  길이 중앙 {st.median([len(x) for x in v]):.0f}")

# 증류 학습셋(JailbreaksOverTime)과 겹치는지
tr = {json.loads(l)["text"] for l in open(HERE/"data_jot/train.jsonl", encoding="utf-8")}
tr |= {json.loads(l)["text"] for l in open(HERE/"data_jot/val.jsonl", encoding="utf-8")}
for k, v in buckets.items():
    print(f"  누수 {k} ∩ 증류학습셋 = {len(set(v) & tr)}건")

rng = random.Random(0)
rows = []
for k, lab in (("adversarial_harmful", 1), ("adversarial_benign", 0)):
    v = buckets[k][:]; rng.shuffle(v)
    for i, t in enumerate(v[:N]):
        rows.append({"uid": f"{lab}_{i}", "prompt": t, "label": lab, "source": k})
rng.shuffle(rows)
json.dump(rows, open(HERE/"wjb_eval_en.json","w"), ensure_ascii=False)
L1=[len(r["prompt"]) for r in rows if r["label"]==1]; L0=[len(r["prompt"]) for r in rows if r["label"]==0]
print(f"\n표본 {len(rows)}건 (공격 {len(L1)} / 정상 {len(L0)})")
print(f"  길이 중앙  공격 {st.median(L1):.0f}  정상 {st.median(L0):.0f}   ← 비슷해야 지름길이 막힘")
print(f"  → wjb_eval_en.json")

"""제3자 한국어 자료로 평가셋 구성. 우리가 만들지 않은 자료라 자체제작 약점을 메운다.

  ① MarkrAI/ko-jailbreak  template 1,138 — 실제 유포된 제일브레이크 템플릿. 영·한 병렬
  ② kimchunsik03/KoreanGuardrail        — 원어민 한국어. hard negative 포함
  ③ xxxjjhhh/korean_guardrail_test  647 — 원어민 한국어
"""
import json, random
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset
HERE=Path(__file__).resolve().parent; rng=random.Random(0)
out=HERE/"data_third"; out.mkdir(exist_ok=True)

# ── ① MarkrAI template : 공격. 정상은 WildChat-ko (영어 정상은 우리 병렬셋에서)
d=load_dataset("MarkrAI/ko-jailbreak")["test"]
tpl=[r for r in d if r["kind"]=="template" and r["prompt_ko"].strip() and r["prompt_en"].strip()]
neg_ko=[json.loads(l)["text"] for l in open(HERE/"wildchat_ko.jsonl",encoding="utf-8")]
neg_ko=list(dict.fromkeys(neg_ko)); rng.shuffle(neg_ko)
va=[{"text":t,"label":0} for t in neg_ko[:1000]]
te=([{"text":r["prompt_ko"],"label":1,"uid":r["id"]} for r in tpl]
    + [{"text":t,"label":0} for t in neg_ko[1000:4000]])
te_en=([{"text":r["prompt_en"],"label":1,"uid":r["id"]} for r in tpl])
for nm,part in (("markr_val",va),("markr_test",te),("markr_test_en",te_en)):
    (out/f"{nm}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in part),encoding="utf-8")
print(f"① MarkrAI  시험 {len(te)} (공격 {len(tpl)}) · 영어판 공격 {len(te_en)}")

# ── ② KoreanGuardrail : attack vs benign_hard_negative
k=load_dataset("kimchunsik03/KoreanGuardrail")["test"]
rows=[{"text":r["text"],"label":int(r["label"]=="attack"),
       "cat":r["category"],"tech":r["technique"]} for r in k if r["text"].strip()]
rows=list({r["text"]:r for r in rows}.values()); rng.shuffle(rows)
nv=[r for r in rows if r["label"]==0]
va2=nv[:500]; s2={v["text"] for v in va2}
te2=[r for r in rows if r["text"] not in s2]
for nm,part in (("kg_val",va2),("kg_test",te2)):
    (out/f"{nm}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in part),encoding="utf-8")
print(f"② KoreanGuardrail 시험 {len(te2)} (공격 {sum(r['label'] for r in te2)})")

# ── ③ korean_guardrail_test
x=load_dataset("xxxjjhhh/korean_guardrail_test")["train"]
r3=[{"text":r["prompt"],"label":int(str(r["is_injection"])=="True")} for r in x if r["prompt"].strip()]
r3=list({r["text"]:r for r in r3}.values()); rng.shuffle(r3)
nv3=[r for r in r3 if r["label"]==0]
va3=nv3[:80]; s3={v["text"] for v in va3}
te3=[r for r in r3 if r["text"] not in s3]
for nm,part in (("xj_val",va3),("xj_test",te3)):
    (out/f"{nm}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in part),encoding="utf-8")
print(f"③ korean_guardrail 시험 {len(te3)} (공격 {sum(r['label'] for r in te3)})")

# 학습셋 누수 확인
tr={json.loads(l)["text"] for l in open(HERE/"data_jot/train.jsonl",encoding="utf-8")}
trk={r["text_ko"] for r in json.load(open(HERE/"jot_train_ko.json",encoding="utf-8"))}
for nm in ("markr_test","kg_test","xj_test"):
    ev={json.loads(l)["text"] for l in open(out/f"{nm}.jsonl",encoding="utf-8")}
    print(f"  누수 {nm}: 영어학습 {len(tr&ev)}건 · 한국어병렬 {len(trk&ev)}건")

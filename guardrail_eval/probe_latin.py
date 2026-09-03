"""가설: 번역문에 남은 영어 트리거 단어 때문에 가드가 '한국어를 이해한 것처럼' 보인다.

검사: (a) 라틴 문자 잔존률  (b) 라틴 문자를 지우면 성능이 무너지는가 (단일 변수)
"""
import json, re
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
J = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
va, te = J("data_koeval/val.jsonl"), J("data_koeval/test.jsonl")
LAT = re.compile(r"[A-Za-z]")
HAN = re.compile(r"[가-힣]")

pos = [r for r in te if r["label"] == 1]; neg = [r for r in te if r["label"] == 0]
def lat_ratio(s):
    h, l = len(HAN.findall(s)), len(LAT.findall(s))
    return l/(h+l) if h+l else 0.0
import statistics as st
print("[a] 라틴 문자 잔존률 (라틴/(한글+라틴))")
for nm, rs in (("한국어 공격 289", pos), ("한국어 정상 4000", neg)):
    v = [lat_ratio(r["text"]) for r in rs]
    print(f"  {nm:16} 중앙 {st.median(v):.3f}  평균 {st.mean(v):.3f}  "
          f">0.2인 비율 {sum(x>0.2 for x in v)/len(v)*100:.1f}%")

TRIG = ["jailbreak","dan","nsfw","chatgpt","aim","developer mode","dude","stan",
        "openai","gpt","ignore","prompt","system"]
hit = {t: sum(1 for r in pos if t in r["text"].lower()) for t in TRIG}
print("\n[b] 공격문에 그대로 남은 영어 트리거 (289건 중)")
for k,v in sorted(hit.items(), key=lambda x:-x[1]):
    if v: print(f"  {k:16} {v:>4}건 ({v/len(pos)*100:.1f}%)")

# (c) 라틴 문자를 전부 지우고 재평가 -- 단일 변수
strip = lambda s: re.sub(r"[A-Za-z]+", " ", s)
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
print("\n[c] 라틴 문자 제거 전후 (같은 모델·같은 임계 절차)")
print(f"{'모델':30} {'원본':>9} {'라틴제거':>9} {'변화':>9}")
for mid, nm in ((str(HERE/"models/jot_monox_kd/best"), "학생 (mDeBERTa)"),
                ("meta-llama/Llama-Prompt-Guard-2-86M", "교사 Prompt Guard 2")):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForSequenceClassification.from_pretrained(mid).cuda().eval().half()
    def sc(texts, bs=64):
        o=[]
        for i in range(0,len(texts),bs):
            e=tok(texts[i:i+bs],truncation=True,max_length=512,padding=True,
                  return_tensors="pt").to("cuda")
            with torch.no_grad(): p=torch.softmax(m(**e).logits.float(),-1)
            o += (1-p[:,0]).tolist()
        return o
    out=[]
    for f in (lambda s: s, strip):
        vs = sc([f(r["text"]) for r in va]); vy=[r["label"] for r in va]
        ts = sc([f(r["text"]) for r in te]); ty=[r["label"] for r in te]
        thr = pick_threshold(vs, vy, 0.01); r,_ = rates(ts, ty, thr); out.append(r)
    print(f"{nm:30} {out[0]:>9.4f} {out[1]:>9.4f} {out[1]-out[0]:>+9.4f}")
    del m; torch.cuda.empty_cache()

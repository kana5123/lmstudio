"""과제 정의를 바꿔서 교사 후보 재평가.

라벨 = adversarial (지시/안전장치를 덮어쓰려는 시도) — Prompt Guard 2 의 과제 정의
     vs prompt_harm_label (유해 콘텐츠) — 다른 층위
"""
import json
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
J = lambda f: [json.loads(l) for l in open(HERE/"data_pgp"/f, encoding="utf-8")]

def rates(s, y, thr):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=thr for a in p)/len(p), sum(a>=thr for a in n)/len(n)

for mid, nm in (("meta-llama/Llama-Prompt-Guard-2-86M", "Prompt Guard 2 86M"),
                ("meta-llama/Prompt-Guard-86M",         "Prompt Guard v1 86M")):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForSequenceClassification.from_pretrained(mid).cuda().eval().half()
    def score(rows, bs=64):
        out=[]
        for i in range(0, len(rows), bs):
            e = tok([r["text"] for r in rows[i:i+bs]], truncation=True, max_length=512,
                    padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out += (1 - torch.softmax(m(**e).logits.float(), -1)[:, 0]).tolist()
        return out
    print(f"\n{'='*72}\n{nm}\n{'='*72}")
    for key, kn in (("adversarial", "지시 덮어쓰기 시도"), ("label", "유해 콘텐츠")):
        print(f"  [라벨 = {kn}]")
        for tag, ln in (("en", "영어"), ("ko", "한국어")):
            va, te = J(f"{tag}_val.jsonl"), J(f"{tag}_test.jsonl")
            vy = [int(r[key]) for r in va]; ty = [int(r[key]) for r in te]
            vs, ts = score(va), score(te)
            thr = pick_threshold(vs, vy, 0.01)
            r1, f1 = rates(ts, ty, thr); r0, f0 = rates(ts, ty, 0.5)
            print(f"    {ln:4} 공격 {sum(ty):>4}/{len(ty)}   "
                  f"기본 {r0:.4f} (FPR {f0*100:4.1f}%)   @1%FPR {r1:.4f} (FPR {f1*100:.2f}%)")
    del m; torch.cuda.empty_cache()

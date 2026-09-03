"""MultiJail 한국어(ICLR 2024, 제3자 자료)로 평가 — 범위 확인용."""
import json, sys, torch
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import BENCHES, pick_threshold
HERE=Path(__file__).resolve().parent
sp=BENCHES["multijail_ko"]()
va,te=sp["val"],sp["test"]
print(f"MultiJail 한국어 · 검증 {len(va)} / 시험 {len(te)} (공격 {sum(y for _,y in te)})\n")
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
print(f"{'모델':30} {'기본(0.5)':>10} {'@1%FPR':>9} {'@0.1%FPR':>10}")
for spec in sys.argv[1:]:
    mp = spec if "/" in spec and not (HERE/"models"/spec).exists() else str(HERE/"models"/spec/"best")
    try:
        tok=AutoTokenizer.from_pretrained(mp)
        m=AutoModelForSequenceClassification.from_pretrained(mp).cuda().eval().half()
    except Exception as e:
        print(f"{spec:30} ✗ {str(e)[:50]}"); continue
    def sc(rows,bs=32):
        o=[]
        for i in range(0,len(rows),bs):
            e=tok([t for t,_ in rows[i:i+bs]],truncation=True,max_length=512,
                  padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad(): o+=(1-torch.softmax(m(**e).logits.float(),-1)[:,0]).tolist()
        return o
    vs,vy=sc(va),[y for _,y in va]; ts,ty=sc(te),[y for _,y in te]
    r0,_=rates(ts,ty,0.5); out=[r0]
    for t in (0.01,0.001):
        thr=pick_threshold(vs,vy,t); r,_=rates(ts,ty,thr); out.append(r)
    nm=spec.split("/")[-1]
    print(f"{nm:30} {out[0]:>10.4f} {out[1]:>9.4f} {out[2]:>10.4f}")
    del m; torch.cuda.empty_cache()

"""학회 자료 평가셋(PIArena ACL2026 + NotInject ACL2025)에서 영·한 병렬 평가."""
import json, math, re, random, sys
from collections import Counter
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold
HERE=Path(__file__).resolve().parent
J=lambda p:[json.loads(l) for l in open(HERE/p,encoding="utf-8")]
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
HITS={}
print(f"{'모델':34} {'영어@1%':>9} {'한국어@1%':>10} {'격차':>8} {'한@0.1%':>9} {'NotInject 오탐':>14}")
for spec in sys.argv[1:]:
    mp = spec if "/" in spec and not (HERE/"models"/spec).exists() else str(HERE/"models"/spec/"best")
    try:
        tok=AutoTokenizer.from_pretrained(mp)
        m=AutoModelForSequenceClassification.from_pretrained(mp).cuda().eval().half()
    except Exception as e: print(f"{spec:34} ✗ {str(e)[:40]}"); continue
    def sc(rs,bs=32):
        o=[]
        for i in range(0,len(rs),bs):
            e=tok([r["text"] for r in rs[i:i+bs]],truncation=True,max_length=512,
                  padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad(): o+=(1-torch.softmax(m(**e).logits.float(),-1)[:,0]).tolist()
        return o
    row=[]; ni_fp=None
    for lang in ("en","ko"):
        va,te=J(f"data_acl/{lang}_val.jsonl"),J(f"data_acl/{lang}_test.jsonl")
        vs,vy=sc(va),[r["label"] for r in va]; ts,ty=sc(te),[r["label"] for r in te]
        thr=pick_threshold(vs,vy,0.01); r1,_=rates(ts,ty,thr); row.append(r1)
        if lang=="ko":
            thr01=pick_threshold(vs,vy,0.001); r01,_=rates(ts,ty,thr01); row.append(r01)
            ni=[(s,r) for s,r in zip(ts,te) if r["src"]=="notinject"]
            ni_fp=sum(s>=thr for s,_ in ni)/len(ni)
            HITS[spec]=[s>=thr for s,r in zip(ts,te) if r["label"]==1]
    nm=spec.split("/")[-1]
    print(f"{nm:34} {row[0]:>9.4f} {row[1]:>10.4f} {row[1]-row[0]:>+8.4f} {row[2]:>9.4f} {ni_fp*100:>13.2f}%")
    del m; torch.cuda.empty_cache()
json.dump({k:v for k,v in HITS.items()}, open(HERE/"results"/"acl_hits.json","w"))

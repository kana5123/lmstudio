"""제일브레이크 주 평가: 영·한 병렬 + 출처별 분해 + NotInject 오탐."""
import json, sys, torch
from collections import Counter
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold
HERE=Path(__file__).resolve().parent
J=lambda p:[json.loads(l) for l in open(HERE/p,encoding="utf-8")]
def rec(s,y,t): 
    p=[a for a,b in zip(s,y) if b==1]; return sum(a>=t for a in p)/len(p) if p else float('nan')
def fpr(s,y,t):
    n=[a for a,b in zip(s,y) if b==0]; return sum(a>=t for a in n)/len(n) if n else float('nan')
HITS={}
hdr=f"{'모델':30} {'영@1%':>8} {'한@1%':>8} {'격차':>8} {'한@0.1%':>8} | {'JOT':>7} {'ITW':>7} {'WJB':>7} | {'NI오탐':>7}"
print(hdr); print("-"*len(hdr))
for spec in sys.argv[1:]:
    mp = spec if "/" in spec and not (HERE/"models"/spec).exists() else str(HERE/"models"/spec/"best")
    try:
        tok=AutoTokenizer.from_pretrained(mp)
        m=AutoModelForSequenceClassification.from_pretrained(mp).cuda().eval().half()
    except Exception as e: print(f"{spec:30} ✗ {str(e)[:40]}"); continue
    def sc(rs,bs=32):
        o=[]
        for i in range(0,len(rs),bs):
            e=tok([r["text"] for r in rs[i:i+bs]],truncation=True,max_length=512,
                  padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad(): o+=(1-torch.softmax(m(**e).logits.float(),-1)[:,0]).tolist()
        return o
    row={}
    for lang in ("en","ko"):
        va,te=J(f"data_jb/{lang}_val.jsonl"),J(f"data_jb/{lang}_test.jsonl")
        vs,vy=sc(va),[r["label"] for r in va]; ts,ty=sc(te),[r["label"] for r in te]
        thr=pick_threshold(vs,vy,0.01); row[lang]=rec(ts,ty,thr)
        if lang=="ko":
            thr01=pick_threshold(vs,vy,0.001); row["ko01"]=rec(ts,ty,thr01)
            for s in ("jot","in_the_wild","wildjailbreak"):
                sub=[(a,r) for a,r in zip(ts,te) if r["src"]==s and r["label"]==1]
                row[s]=sum(a>=thr for a,_ in sub)/len(sub)
            nis=[(a,r) for a,r in zip(ts,te) if r["src"]=="notinject"]
            row["ni"]=sum(a>=thr for a,_ in nis)/len(nis)
            HITS[spec]=[bool(a>=thr) for a,r in zip(ts,te) if r["label"]==1]
    nm=spec.split("/")[-1]
    print(f"{nm:30} {row['en']:>8.4f} {row['ko']:>8.4f} {row['ko']-row['en']:>+8.4f} {row['ko01']:>8.4f} | "
          f"{row['jot']:>7.4f} {row['in_the_wild']:>7.4f} {row['wildjailbreak']:>7.4f} | {row['ni']*100:>6.1f}%")
    del m; torch.cuda.empty_cache()
json.dump(HITS, open(HERE/"results"/"jb_hits.json","w"))

"""최종 표: 영·한 병렬 평가 + 어휘 대조군."""
import json, math, re, random
from collections import Counter
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold
HERE = Path(__file__).resolve().parent
J = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)

MODELS=[(str(HERE/"models/jot_monox_kd/best"),"학생 mDeBERTa (증류·한국어 0건)"),
        ("meta-llama/Llama-Prompt-Guard-2-86M","교사 Prompt Guard 2"),
        ("meta-llama/Prompt-Guard-86M","Prompt Guard v1"),
        ("protectai/deberta-v3-base-prompt-injection-v2","ProtectAI v2"),
        ("leolee99/PIGuard","PIGuard (ACL 2025)")]
res={}
for mid,nm in MODELS:
    try:
        trc = "PIGuard" in mid
        tok=AutoTokenizer.from_pretrained(mid, trust_remote_code=trc)
        m=AutoModelForSequenceClassification.from_pretrained(mid, trust_remote_code=trc).cuda().eval().half()
    except Exception as e:
        print(f"{nm}: ✗ {str(e)[:60]}"); continue
    def sc(rs,bs=32):
        o=[]
        for i in range(0,len(rs),bs):
            e=tok([r["text"] for r in rs[i:i+bs]],truncation=True,max_length=512,
                  padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad(): p=torch.softmax(m(**e).logits.float(),-1)
            o+=(1-p[:,0]).tolist()
        return o
    row={}
    for lang in ("en","ko"):
        va,te=J(f"data_style/{lang}_val.jsonl"),J(f"data_style/{lang}_test.jsonl")
        vs,vy=sc(va),[r["label"] for r in va]; ts,ty=sc(te),[r["label"] for r in te]
        d={}
        d["기본"],_=rates(ts,ty,0.5)
        for t,k in ((0.01,"1"),(0.001,"0.1")):
            thr=pick_threshold(vs,vy,t); d[k],_=rates(ts,ty,thr)
        row[lang]=d
    res[nm]=row; del m; torch.cuda.empty_cache()
    print(f"{nm} 완료", flush=True)

# 어휘 대조군
def lex(lang):
    va,te=J(f"data_style/{lang}_val.jsonl"),J(f"data_style/{lang}_test.jsonl")
    rng=random.Random(0); idx=list(range(len(te))); rng.shuffle(idx); h=len(idx)//2
    tr=[te[i] for i in idx[:h]]; ev=[te[i] for i in idx[h:]]
    def F(s,n=3,cap=3000):
        s=re.sub(r"\s+"," ",s.strip())[:cap]
        return Counter(s[i:i+n] for i in range(max(len(s)-n+1,0)))
    df=Counter(); trf=[F(r["text"]) for r in tr]
    for f in trf: df.update(f.keys())
    vocab={g:i for i,(g,c) in enumerate(df.most_common(20000)) if c>=3}
    N=len(tr); idf={g:math.log(N/(1+df[g])) for g in vocab}
    def V(f):
        v=torch.zeros(len(vocab)); t=sum(f.values()) or 1
        for g,c in f.items():
            j=vocab.get(g)
            if j is not None: v[j]=(c/t)*idf[g]
        return v
    X=torch.stack([V(f) for f in trf]); y=torch.tensor([r["label"] for r in tr]).float()
    Xe=torch.stack([V(F(r["text"])) for r in ev]); ye=[r["label"] for r in ev]
    Xv=torch.stack([V(F(r["text"])) for r in va]); yv=[r["label"] for r in va]
    head=nn.Linear(X.shape[1],1)
    opt=torch.optim.AdamW(head.parameters(),lr=5e-2,weight_decay=1e-3)
    lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/max((y==1).sum(),1)))
    for _ in range(600): opt.zero_grad(); lf(head(X).squeeze(-1),y).backward(); opt.step()
    with torch.no_grad():
        sv=torch.sigmoid(head(Xv).squeeze(-1)).tolist(); se=torch.sigmoid(head(Xe).squeeze(-1)).tolist()
    d={}; d["기본"],_=rates(se,ye,0.5)
    for t,k in ((0.01,"1"),(0.001,"0.1")):
        thr=pick_threshold(sv,yv,t); d[k],_=rates(se,ye,thr)
    return d
res["어휘 대조군 (문자 3-gram)"]={l:lex(l) for l in ("en","ko")}

print(f"\n{'='*86}")
print(f"{'모델':34} {'영어 @1%':>9} {'한국어 @1%':>11} {'격차':>8} {'영 @0.1%':>9} {'한 @0.1%':>9}")
print("="*86)
for nm,r in res.items():
    print(f"{nm:34} {r['en']['1']:>9.4f} {r['ko']['1']:>11.4f} "
          f"{r['ko']['1']-r['en']['1']:>+8.4f} {r['en']['0.1']:>9.4f} {r['ko']['0.1']:>9.4f}")
(HERE/"results"/"final_style.json").write_text(json.dumps(res,ensure_ascii=False,indent=1),encoding="utf-8")

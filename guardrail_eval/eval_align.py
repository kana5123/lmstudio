"""정렬 손실 층별 결과: 한국어 탐지 성능 + 12층 정렬도 회복 여부."""
import json, sys, torch, torch.nn.functional as F
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold
HERE=Path(__file__).resolve().parent
J=lambda p:[json.loads(l) for l in open(HERE/p,encoding="utf-8")]
sim=json.load(open(HERE/"neardup_sim.json")); TH=0.7
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
# 층별 정렬도 재측정용 병렬 문장
PEN=[json.loads(l)["text"] for l in open(HERE/"data_par/en_test.jsonl",encoding="utf-8")][:600]
PKO=[json.loads(l)["text"] for l in open(HERE/"data_par/ko_test.jsonl",encoding="utf-8")][:600]

print(f"{'모델':22} {'한국어 전체':>10} {'한국어 신규':>10} {'영어 신규':>9} {'정렬 L8':>8} {'정렬 L12':>9}")
for name in sys.argv[1:]:
    mp=HERE/"models"/name/"best"
    if not (mp/"model.safetensors").exists(): print(f"{name:22} (미완료)"); continue
    tok=AutoTokenizer.from_pretrained(mp)
    m=AutoModelForSequenceClassification.from_pretrained(mp,output_hidden_states=True).cuda().eval().half()
    def sc(rs,bs=32):
        o=[]
        for i in range(0,len(rs),bs):
            e=tok([r["text"] for r in rs[i:i+bs]],truncation=True,max_length=512,
                  padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad(): o+=(1-torch.softmax(m(**e).logits.float(),-1)[:,0]).tolist()
        return o
    res=[]
    for lang in ("ko","en"):
        va,te=J(f"data_par/{lang}_val.jsonl"),J(f"data_par/{lang}_test.jsonl")
        thr=pick_threshold(sc(va),[r["label"] for r in va],0.01)
        sel_all=te; sel_nov=[r for r in te if r["label"]==0 or sim.get(r["uid"],0)<TH]
        for sel in ((sel_all,sel_nov) if lang=="ko" else (sel_nov,)):
            ts,ty=sc(sel),[r["label"] for r in sel]; r,_=rates(ts,ty,thr); res.append(r)
    # 정렬도
    def emb(ts,layer,bs=24):
        o=[]
        for i in range(0,len(ts),bs):
            e=tok(ts[i:i+bs],truncation=True,max_length=256,padding=True,
                  return_tensors="pt").to("cuda")
            with torch.no_grad(): h=m(**e,output_hidden_states=True).hidden_states[layer].float()
            msk=e["attention_mask"].unsqueeze(-1).float()
            o.append(F.normalize(((h*msk).sum(1)/msk.sum(1)),dim=-1).cpu())
        return torch.cat(o)
    al=[]
    for L in (8,12):
        A,B=emb(PEN,L),emb(PKO,L)
        al.append(((B@A.T).argmax(1)==torch.arange(len(B))).float().mean().item())
    print(f"{name:22} {res[0]:>10.4f} {res[1]:>10.4f} {res[2]:>9.4f} {al[0]*100:>7.2f}% {al[1]*100:>8.2f}%")
    del m; torch.cuda.empty_cache()

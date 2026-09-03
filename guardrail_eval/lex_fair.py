"""공정한 어휘 대조군: 학생과 '똑같은 조건'으로 학습시킨다.

앞선 대조군은 시험셋 절반으로 학습해서 유리했다. 학생은
  · 영어 2023-02~11 로만 학습
  · 한국어는 한 글자도 못 봄
  · 2023-12 (미래 공격) 로 평가
대조군도 같은 조건이어야 비교가 성립한다.
"""
import json, math, re
from collections import Counter
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from rfpr import pick_threshold
HERE = Path(__file__).resolve().parent
J = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
def rates(s,y,t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
def F(s,n=3,cap=3000):
    s=re.sub(r"\s+"," ",s.strip())[:cap]
    return Counter(s[i:i+n] for i in range(max(len(s)-n+1,0)))

# 학습: 학생과 동일한 영어 데이터
tr = J("data_jot/train.jsonl")
print(f"대조군 학습 {len(tr)}건 (영어 2023-02~11, 학생과 동일)")
df=Counter(); trf=[F(r["text"]) for r in tr]
for f in trf: df.update(f.keys())
vocab={g:i for i,(g,c) in enumerate(df.most_common(30000)) if c>=5}
N=len(tr); idf={g:math.log(N/(1+df[g])) for g in vocab}
def V(f):
    v=torch.zeros(len(vocab)); t=sum(f.values()) or 1
    for g,c in f.items():
        j=vocab.get(g)
        if j is not None: v[j]=(c/t)*idf[g]
    return v
X=torch.stack([V(f) for f in trf]); y=torch.tensor([r["label"] for r in tr]).float()
head=nn.Linear(X.shape[1],1)
opt=torch.optim.AdamW(head.parameters(),lr=5e-2,weight_decay=1e-3)
lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/(y==1).sum()))
for _ in range(800): opt.zero_grad(); lf(head(X).squeeze(-1),y).backward(); opt.step()
print(f"  어휘 {len(vocab)}개, 학습 완료\n")

print(f"{'평가':38} {'기본':>8} {'@1%FPR':>9} {'@0.1%FPR':>9}")
for lang, nm in (("en","영어 2023-12 (미래 공격)"), ("ko","한국어 2023-12 (번역)")):
    va,te=J(f"data_par/{lang}_val.jsonl"),J(f"data_par/{lang}_test.jsonl")
    with torch.no_grad():
        sv=torch.sigmoid(head(torch.stack([V(F(r["text"])) for r in va])).squeeze(-1)).tolist()
        ts=torch.sigmoid(head(torch.stack([V(F(r["text"])) for r in te])).squeeze(-1)).tolist()
    vy=[r["label"] for r in va]; ty=[r["label"] for r in te]
    r0,_=rates(ts,ty,0.5); out=[r0]
    for t in (0.01,0.001):
        thr=pick_threshold(sv,vy,t); r,_=rates(ts,ty,thr); out.append(r)
    print(f"{nm:38} {out[0]:>8.4f} {out[1]:>9.4f} {out[2]:>9.4f}")

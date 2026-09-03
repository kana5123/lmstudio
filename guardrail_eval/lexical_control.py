"""어휘 대조군: 문자 n-gram TF-IDF + 선형분류기가 몇 점 내는가.

다른 세션 기록(jot_tpfp_corpus_confound): JailbreaksOverTime 의 공격/정상 구분이
'말뭉치 판별'일 수 있고, TF-IDF 대조군이 0.9911 을 냈다. 우리 한국어 평가셋도
같은 구조(공격=jailbreak_chat/flowgpt 번역, 정상=WildChat-ko)이므로 반드시 확인한다.

모델이 대조군을 못 넘으면, 그 성능은 '제일브레이크 탐지'가 아니라 '출처 판별'이다.
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
va, te = J("data_koeval/val.jsonl"), J("data_koeval/test.jsonl")

# 학습용: 시험셋을 반으로 갈라 대조군을 '학습'시킨다 (대조군에 유리하게 -- 상한을 재려는 것)
import random
rng = random.Random(0)
idx = list(range(len(te))); rng.shuffle(idx)
half = len(idx)//2
tr_rows = [te[i] for i in idx[:half]]; ev_rows = [te[i] for i in idx[half:]]
print(f"대조군 학습 {len(tr_rows)} / 평가 {len(ev_rows)} (시험셋을 반으로 나눔)")
print(f"  평가 쪽 공격 {sum(r['label'] for r in ev_rows)}건\n")

def feats(s, n=3, cap=3000):
    s = re.sub(r"\s+", " ", s.strip())[:cap]
    return Counter(s[i:i+n] for i in range(max(len(s)-n+1, 0)))

# 어휘 사전 (학습 쪽에서만)
df = Counter()
tr_f = [feats(r["text"]) for r in tr_rows]
for f in tr_f: df.update(f.keys())
vocab = {g: i for i, (g, c) in enumerate(df.most_common(20000)) if c >= 3}
N = len(tr_rows)
idf = {g: math.log(N/(1+df[g])) for g in vocab}
print(f"문자 3-gram 어휘 {len(vocab)}개")

def vec(f):
    v = torch.zeros(len(vocab))
    tot = sum(f.values()) or 1
    for g, c in f.items():
        j = vocab.get(g)
        if j is not None: v[j] = (c/tot) * idf[g]
    return v

X = torch.stack([vec(f) for f in tr_f]); y = torch.tensor([r["label"] for r in tr_rows]).float()
Xe = torch.stack([vec(feats(r["text"])) for r in ev_rows]); ye = [r["label"] for r in ev_rows]
Xv = torch.stack([vec(feats(r["text"])) for r in va]);      yv = [r["label"] for r in va]

head = nn.Linear(X.shape[1], 1)
opt = torch.optim.AdamW(head.parameters(), lr=5e-2, weight_decay=1e-3)
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/(y==1).sum()))
for _ in range(600):
    opt.zero_grad(); lossf(head(X).squeeze(-1), y).backward(); opt.step()
with torch.no_grad():
    sv = torch.sigmoid(head(Xv).squeeze(-1)).tolist()
    se = torch.sigmoid(head(Xe).squeeze(-1)).tolist()

def rates(s, yy, t):
    p=[a for a,b in zip(s,yy) if b==1]; n=[a for a,b in zip(s,yy) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)
# AUROC
def auroc(s, yy):
    pairs = sorted(zip(s, yy)); r = {}; 
    ranks = [0]*len(s)
    order = sorted(range(len(s)), key=lambda i: s[i])
    for k, i in enumerate(order): ranks[i] = k+1
    P = sum(yy); Nn = len(yy)-P
    sr = sum(ranks[i] for i in range(len(s)) if yy[i]==1)
    return (sr - P*(P+1)/2) / (P*Nn)

print(f"\n어휘 대조군 (문자 3-gram TF-IDF + 선형)")
print(f"  AUROC {auroc(se, ye):.4f}")
for t in (0.01, 0.001):
    thr = pick_threshold(sv, yv, t)
    r, f = rates(se, ye, thr)
    print(f"  @{t*100:g}%FPR  Recall {r:.4f}  달성 FPR {f*100:.2f}%")

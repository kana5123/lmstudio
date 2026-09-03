"""어느 층에서 판정해야 영·한 격차가 가장 작은가.

영어로만 학습한 모델의 층별 표현 위에, 영어 데이터로만 선형 분류기를 얹고
영어/한국어 시험셋을 각각 평가한다. 한국어는 학습에 전혀 안 들어간다.
"""
import json, sys
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(HERE/"models"/"mdeberta_en_safety"/"best")
N_TRAIN = 6000
tok = AutoTokenizer.from_pretrained(CKPT)
enc = AutoModel.from_pretrained(CKPT, output_hidden_states=True).cuda().eval().half()

def feats(texts, bs=48):
    acc = None
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=256,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = enc(**e).hidden_states
        m = e["attention_mask"].unsqueeze(-1).float()
        v = [((h.float()*m).sum(1)/m.sum(1)).cpu() for h in hs]
        acc = v if acc is None else [torch.cat([a, b]) for a, b in zip(acc, v)]
    return acc

J = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
tr = J("data_wg/train.jsonl")[:N_TRAIN]
va, ete, kte = J("data_pgp/en_val.jsonl"), J("data_pgp/en_test.jsonl"), J("data_pgp/ko_test.jsonl")
print(f"탐침 학습 {len(tr)} (영어) · 임계 {len(va)} (영어) · 시험 {len(ete)}/{len(kte)}")

T = feats([r["text"] for r in tr]);  ty = torch.tensor([r["label"] for r in tr]).float()
V = feats([r["text"] for r in va]);  vy = [r["label"] for r in va]
E = feats([r["text"] for r in ete]); ey = [r["label"] for r in ete]
K = feats([r["text"] for r in kte]); ky = [r["label"] for r in kte]

def rates(s, y, thr):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=thr for a in p)/len(p), sum(a>=thr for a in n)/len(n)

print(f"\n{'층':>4} {'영어':>9} {'한국어':>9} {'격차':>9}   (Recall @1%FPR)")
best = None
for li in range(len(T)):
    x = T[li]; mu, sd = x.mean(0), x.std(0) + 1e-6
    nz = lambda z: (z - mu) / sd
    head = nn.Linear(x.shape[1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt = nz(x)
    for _ in range(400):
        opt.zero_grad(); lossf(head(xt).squeeze(-1), ty).backward(); opt.step()
    with torch.no_grad():
        sc = lambda z: torch.sigmoid(head(nz(z)).squeeze(-1)).tolist()
        thr = pick_threshold(sc(V[li]), vy, 0.01)
        er, _ = rates(sc(E[li]), ey, thr)
        kr, _ = rates(sc(K[li]), ky, thr)
    mark = ""
    if best is None or kr > best[1]: best = (li, kr); mark = " ←"
    print(f"{li:>4} {er:>9.4f} {kr:>9.4f} {er-kr:>+9.4f}{mark}")
print(f"\n★ 한국어 최고: {best[0]}층  {best[1]:.4f}")

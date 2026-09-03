"""번역 품질 검사: 영·한 교차언어 임베딩 유사도 + 규칙 기반 이상 탐지.

bge-m3 는 영어와 한국어를 같은 공간에 놓으므로, 같은 뜻이면 코사인이 높다.
대조군으로 '짝이 아닌 쌍'의 유사도를 같이 재서 기준선을 만든다.
"""
import json, re, statistics as st
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
rows = json.load(open(HERE/"jot_2023_12_ko_flagged.json", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
m = AutoModel.from_pretrained("BAAI/bge-m3").cuda().eval().half()

def emb(ts, bs=16):
    o=[]
    for i in range(0,len(ts),bs):
        e=tok(ts[i:i+bs],truncation=True,max_length=512,padding=True,
              return_tensors="pt").to("cuda")
        with torch.no_grad(): h=m(**e).last_hidden_state[:,0]   # CLS
        o.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(o)

E = emb([r["prompt"] for r in rows]); K = emb([r["text_ko"] for r in rows])
same = (E*K).sum(-1)
shuf = (E*K[torch.randperm(len(K), generator=torch.Generator().manual_seed(0))]).sum(-1)
print(f"영↔한 교차언어 유사도 (bge-m3)")
print(f"  짝 맞는 쌍   중앙 {same.median():.4f}  평균 {same.mean():.4f}  최소 {same.min():.4f}")
print(f"  짝 아닌 쌍   중앙 {shuf.median():.4f}  평균 {shuf.mean():.4f}")
print(f"  → 변별력 {same.mean()-shuf.mean():+.4f}")

for th in (0.9, 0.85, 0.8, 0.75, 0.7):
    print(f"  유사도 < {th}: {int((same<th).sum())}건 ({(same<th).float().mean()*100:.1f}%)")

for i, r in enumerate(rows): r["_sim"] = round(float(same[i]), 4)
low = sorted(rows, key=lambda r: r["_sim"])[:12]
print(f"\n유사도 최저 12건:")
for r in low:
    print(f"  {r['_sim']:.3f} [{r['_ratio']}x, 반복{r['_rep']}, 영{r['_eng']}] "
          f"{r['prompt'][:60].replace(chr(10),' ')}")
    print(f"        → {r['text_ko'][:60].replace(chr(10),' ')}")
json.dump(rows, open(HERE/"jot_2023_12_ko_flagged.json","w"), ensure_ascii=False, indent=1)
print(f"\n저장 (유사도 추가) → jot_2023_12_ko_flagged.json")

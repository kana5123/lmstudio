"""측정 1·2 를 실제 문장으로 보여준다."""
import json
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CKPT = HERE / "models" / "mdeberta_en_guard" / "best"
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(
    CKPT, output_hidden_states=True).cuda().eval().half()

def run(texts, bs=64):
    embs, risks = [], []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=512,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**e)
        m = e["attention_mask"].unsqueeze(-1).float()
        h = o.hidden_states[-1].float()
        embs.append(((h*m).sum(1)/m.sum(1)).cpu())
        risks.append(torch.softmax(o.logits.float(), -1)[:, 1].cpu())
    return torch.cat(embs), torch.cat(risks)

L = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
en, ko = L("data_en/test.jsonl"), L("data_ko/test.jsonl")
E, RE = run([r["text"] for r in en])
K, RK = run([r["text"] for r in ko])

print("="*78)
print("측정 1 — 같은 문장을 영어로 줬을 때 vs 한국어로 줬을 때의 위험 점수")
print("="*78)
for i in (5, 12, 30):
    print(f"\n[{i}번 기록]  정답 라벨: {'공격' if en[i]['label'] else '정상'}")
    print(f"  영어  : {en[i]['text'][:88]}")
    print(f"  한국어: {ko[i]['text'][:70]}")
    print(f"  ▸ 위험점수  영어 {RE[i]:.4f}   한국어 {RK[i]:.4f}   차이 {abs(RE[i]-RK[i]):.4f}")

print("\n" + "="*78)
print("측정 2 — 한국어 문장 하나를 주고, 영어 1,797개 중 '내 짝'을 찾게 시킴")
print("="*78)
En, Kn = F.normalize(E, dim=-1), F.normalize(K, dim=-1)
i = 12
sims = (Kn[i] @ En.T)
order = sims.argsort(descending=True)
rank = (order == i).nonzero().item() + 1
print(f"\n질의 (한국어 {i}번): {ko[i]['text'][:70]}")
print(f"\n영어 1,797개와의 유사도 상위 5개:")
for r, j in enumerate(order[:5].tolist(), 1):
    mark = "  ← ★진짜 짝" if j == i else ""
    print(f"  {r}위  cos {sims[j]:.4f}  [{j:>4}] {en[j]['text'][:58]}{mark}")
print(f"\n  진짜 짝({i}번)은 cos {sims[i]:.4f} 로 ... {rank}위")
print(f"  → 짝보다 유사도가 높은 '남남' 문장이 {rank-1}개 있음")

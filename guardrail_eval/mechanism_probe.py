"""교차언어 전이가 '어느 층에서' 일어나는지 측정.

같은 뜻의 영/한 문장 1,797쌍을 넣고, 층마다
  (1) 자기 짝을 찾아내는 정확도  (2) 짝 유사도 - 남남 유사도
를 잰다. 다국어 백본과 영어 전용 백본을 같은 방식으로 비교한다.
"""
import json, sys
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
L = lambda p: [json.loads(l)["text"] for l in open(HERE/p, encoding="utf-8")]
EN, KO = L("data_en/test.jsonl"), L("data_ko/test.jsonl")

def layerwise(mid, n=1200):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModel.from_pretrained(mid, output_hidden_states=True).cuda().eval().half()
    def emb(texts, bs=48):
        acc = None
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i+bs], truncation=True, max_length=256,
                    padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                hs = m(**e).hidden_states                       # (층+1) 개
            msk = e["attention_mask"].unsqueeze(-1).float()
            v = [((h.float()*msk).sum(1)/msk.sum(1)).cpu() for h in hs]
            acc = v if acc is None else [torch.cat([a, b]) for a, b in zip(acc, v)]
        return acc
    A, B = emb(EN[:n]), emb(KO[:n])
    out = []
    for li, (a, b) in enumerate(zip(A, B)):
        an, bn = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        S = bn @ an.T
        acc = (S.argmax(1) == torch.arange(len(b))).float().mean().item()
        same = S.diag().mean().item()
        other = (S.sum() - S.diag().sum()).item() / (S.numel() - len(S))
        out.append((li, acc, same, other, same - other))
    del m; torch.cuda.empty_cache()
    return out

for mid, nm in ((sys.argv[1] if len(sys.argv)>1 else "microsoft/mdeberta-v3-base", "다국어 백본"),
                ("microsoft/deberta-v3-base", "영어 전용 백본")):
    print(f"\n{'='*66}\n{nm}   {mid}\n{'='*66}")
    print(f"{'층':>4} {'짝찾기 정확도':>14} {'짝 유사도':>10} {'남남 유사도':>11} {'변별력':>9}")
    for li, acc, same, other, gap in layerwise(mid):
        bar = "█" * int(acc * 60)
        print(f"{li:>4} {acc*100:>12.2f}% {same:>10.4f} {other:>11.4f} {gap:>9.4f}  {bar}")

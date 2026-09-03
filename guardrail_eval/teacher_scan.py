"""교사 후보를 영어 PolyGuardPrompts 에서 비교. 증류의 천장을 먼저 확인한다."""
import json, sys
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
L = lambda f: [json.loads(l) for l in open(HERE/"data_pgp"/f, encoding="utf-8")]

CANDS = [
 ("leolee99/PIGuard",                            "PIGuard (ACL 2025)",   True),
 ("meta-llama/Llama-Prompt-Guard-2-86M",         "Prompt Guard 2 86M",   False),
 ("meta-llama/Prompt-Guard-86M",                 "Prompt Guard v1 86M",  False),
 ("protectai/deberta-v3-base-prompt-injection-v2","ProtectAI v2",        False),
]

def rates(s, y, thr):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=thr for a in p)/len(p), sum(a>=thr for a in n)/len(n)

va, te = L("en_val.jsonl"), L("en_test.jsonl")
for mid, nm, trc in CANDS:
    try:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=trc)
        m = AutoModelForSequenceClassification.from_pretrained(
            mid, trust_remote_code=trc).cuda().eval().half()
    except Exception as e:
        print(f"{nm:24} ✗ {type(e).__name__}: {str(e)[:80]}"); continue
    lab = {i: v for i, v in getattr(m.config, "id2label", {}).items()}
    # '위험' 확률 = 안전(첫 라벨)이 아닌 라벨들의 확률 합
    def score(rows, bs=64):
        out=[]
        for i in range(0, len(rows), bs):
            e = tok([r["text"] for r in rows[i:i+bs]], truncation=True, max_length=512,
                    padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                p = torch.softmax(m(**e).logits.float(), -1)
            out += (1 - p[:, 0]).tolist()
        return out
    vs, ts = score(va), score(te)
    vy = [r["label"] for r in va]; ty = [r["label"] for r in te]
    thr = pick_threshold(vs, vy, 0.01); rec1, f1 = rates(ts, ty, thr)
    rec0, f0 = rates(ts, ty, 0.5)
    print(f"{nm:24} 라벨={list(lab.values())}")
    print(f"{'':24} 기본 {rec0:.4f} (FPR {f0*100:.1f}%)   @1%FPR {rec1:.4f} (FPR {f1*100:.2f}%)")
    del m; torch.cuda.empty_cache()

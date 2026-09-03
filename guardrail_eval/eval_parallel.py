"""영·한 병렬 평가 (PolyGuardPrompts). 같은 항목이므로 언어 격차만 순수하게 잰다.

임계값은 각 언어의 val 에서 고르고 test 에 적용 (rfpr 프로토콜과 동일).
"""
import json, os, sys
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(HERE/"models"/"mdeberta_en_safety"/"best")
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).cuda().eval().half()

def score(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=512,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out += torch.softmax(model(**e).logits.float(), -1)[:, 1].tolist()
    return out

def rates(s, y, thr):
    p = [a for a, b in zip(s, y) if b == 1]; n = [a for a, b in zip(s, y) if b == 0]
    return sum(a >= thr for a in p)/len(p), sum(a >= thr for a in n)/len(n)

PFX = os.environ.get("PFX", "")   # 과제 전환: PFX=adv_ 면 지시 덮어쓰기 라벨
L = lambda f: [json.loads(l) for l in open(HERE/"data_pgp"/(PFX+f), encoding="utf-8")]
print(f"모델: {CKPT}\n")
res = {}
for tag, nm in (("en", "영어"), ("ko", "한국어")):
    va, te = L(f"{tag}_val.jsonl"), L(f"{tag}_test.jsonl")
    vs, vy = score([r["text"] for r in va]), [r["label"] for r in va]
    ts, ty = score([r["text"] for r in te]), [r["label"] for r in te]
    r = {}
    rec, fpr = rates(ts, ty, 0.5); r["기본(0.5)"] = (rec, fpr, None)
    for t in (0.01, 0.001):
        thr = pick_threshold(vs, vy, t)
        rec, fpr = rates(ts, ty, thr); r[f"FPR {t*100:g}%"] = (rec, fpr, thr)
    # 적대적 부분집합만
    adv = [(s, y) for s, y, rr in zip(ts, ty, te) if rr.get("adversarial", rr["label"])]
    thr = pick_threshold(vs, vy, 0.01)
    ar, af = rates([a for a, _ in adv], [b for _, b in adv], thr)
    res[tag] = r
    print(f"=== {nm}  시험 {len(te)}건 (유해 {sum(ty)})")
    for k, (rec, fpr, thr) in r.items():
        ex = f"  임계 {thr:.6f}" if thr else ""
        print(f"  {k:10} Recall {rec:.4f}  달성 FPR {fpr*100:.2f}%{ex}")
    print(f"  {'적대적만':10} Recall {ar:.4f}  ({len(adv)}건, @1%FPR 임계)")

g = res["en"]["FPR 1%"][0] - res["ko"]["FPR 1%"][0]
print(f"\n★ 영어 → 한국어 격차 @1%FPR : {res['en']['FPR 1%'][0]:.4f} → "
      f"{res['ko']['FPR 1%'][0]:.4f}   ({g:+.4f})")

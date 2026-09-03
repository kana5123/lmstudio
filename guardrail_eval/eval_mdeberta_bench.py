"""학습한 mDeBERTa 한국어 가드를 학습 데이터와 무관한 벤치마크에서 평가한다.

임계값은 각 벤치마크의 val 에서만 고르고 test 에 그대로 적용한다 (rfpr.py 와 동일 프로토콜).
"""
import json, os, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rfpr import BENCHES, pick_threshold


def own_split(d):
    """학습에 쓴 자기 분할(분포 안). data_ko / data_en."""
    def load(n):
        return [(r["text"], r["label"]) for r in
                (json.loads(l) for l in open(HERE / d / f"{n}.jsonl", encoding="utf-8"))]
    return {"val": load("val"), "test": load("test")}


BENCHES = dict(BENCHES, own_ko=lambda: own_split("data_ko"),
               own_en=lambda: own_split("data_en"))
HERE = Path(__file__).resolve().parent
RUN = os.environ.get("RUN", "mdeberta_ko_guard")
CKPT = HERE / "models" / RUN / "best"
MAX_LEN, BS = 512, 64


def rates(scores, labels, thr):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    return (sum(s >= thr for s in pos) / len(pos),
            sum(s >= thr for s in neg) / len(neg))


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).cuda().eval().half()

    def score(texts):
        out = []
        for i in range(0, len(texts), BS):
            enc = tok(texts[i:i + BS], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
            out += p.tolist()
        return out

    names = sys.argv[1:] or ["kopi", "multijail_ko"]
    allres = {}
    for name in names:
        sp = BENCHES[name]()
        va, te = sp["val"], sp["test"]
        t0 = time.time()
        vs = score([t for t, _ in va]); vy = [y for _, y in va]
        ts = score([t for t, _ in te]); ty = [y for _, y in te]
        dt = (time.time() - t0) / (len(va) + len(te)) * 1000
        r = {"n_val": len(va), "n_test": len(te),
             "pos_test": sum(ty), "ms_per_sample": round(dt, 2)}
        rec, fpr = rates(ts, ty, 0.5)
        r["기본(0.5)"] = dict(recall=rec, fpr=fpr)
        for tgt in (0.01, 0.001):
            thr = pick_threshold(vs, vy, tgt)
            rec, fpr = rates(ts, ty, thr)
            r[f"FPR {tgt*100:g}%"] = dict(thr=thr, recall=rec, fpr=fpr)
        allres[name] = r
        print(f"\n=== {name}  검증 {len(va)} / 시험 {len(te)} (공격 {sum(ty)})  {dt:.1f}ms/건")
        for k, v in r.items():
            if isinstance(v, dict):
                ex = f"  임계 {v['thr']:.6f}" if "thr" in v else ""
                print(f"  {k:10} Recall {v['recall']:.4f}  달성 FPR {v['fpr']*100:.2f}%{ex}")
    (HERE / "results" / f"mdeberta_cross_bench_{RUN}.json").write_text(
        json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

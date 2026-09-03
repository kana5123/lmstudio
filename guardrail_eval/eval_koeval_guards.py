"""공개 가드들을 우리 한국어 제일브레이크 평가셋에 돌린다."""
import json, sys, time
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from guards import GUARDS
from rfpr import pick_threshold

HERE = Path(__file__).resolve().parent
J = lambda p: [json.loads(l) for l in open(HERE/p, encoding="utf-8")]
va, te = J("data_koeval/val.jsonl"), J("data_koeval/test.jsonl")

def rates(s, y, t):
    p=[a for a,b in zip(s,y) if b==1]; n=[a for a,b in zip(s,y) if b==0]
    return sum(a>=t for a in p)/len(p), sum(a>=t for a in n)/len(n)

name = sys.argv[1]
g = GUARDS[name]()
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
def sc(rows):
    out, t0 = [], time.time()
    for i in range(0, len(rows), B):
        out += [x for x, _ in g.score([r["text"] for r in rows[i:i+B]])]
        if (i//B) % 50 == 0:
            print(f"    {i+B}/{len(rows)} {time.time()-t0:.0f}초", flush=True)
    return out

vs, vy = sc(va), [r["label"] for r in va]
ts, ty = sc(te), [r["label"] for r in te]
res = {"guard": name, "n_test": len(te), "pos_test": sum(ty)}
r0, f0 = rates(ts, ty, 0.5)
res["기본"] = {"recall": round(r0,4), "fpr": round(f0,4)}
for t, k in ((0.01,"1pct"), (0.001,"0.1pct")):
    thr = pick_threshold(vs, vy, t); r, f = rates(ts, ty, thr)
    res[f"@{k}"] = {"thr": thr, "recall": round(r,4), "fpr": round(f,4)}
print(json.dumps(res, ensure_ascii=False))
(HERE/"results"/f"koeval_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")

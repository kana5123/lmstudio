"""번역 자동 검증 + 한국어 평가셋 구성."""
import json, re, statistics as st
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

HERE = Path(__file__).resolve().parent
rows = json.load(open(HERE/"jot_2023_12_ko.json", encoding="utf-8"))
HAN = re.compile(r"[가-힣]"); LAT = re.compile(r"[A-Za-z]")

def eng_ratio(s):
    h, l = len(HAN.findall(s)), len(LAT.findall(s))
    return l / (h + l) if h + l else 1.0

def repeat_score(s):
    """가장 흔한 3어절 덩어리가 전체에서 차지하는 비율. 높으면 반복 붕괴."""
    w = s.split()
    if len(w) < 12: return 0.0
    from collections import Counter
    g = Counter(tuple(w[i:i+3]) for i in range(len(w)-2))
    return g.most_common(1)[0][1] * 3 / len(w)

flags = {}
for r in rows:
    f = []
    ratio = len(r["text_ko"]) / max(len(r["prompt"]), 1)
    if ratio < 0.25: f.append("너무짧음")
    if ratio > 1.5: f.append("너무김")
    if eng_ratio(r["text_ko"]) > 0.5: f.append("영어잔존")
    if repeat_score(r["text_ko"]) > 0.35: f.append("반복붕괴")
    if not HAN.search(r["text_ko"]): f.append("한글없음")
    r["_flags"] = f; r["_ratio"] = round(ratio, 2)
    r["_rep"] = round(repeat_score(r["text_ko"]), 2)
    r["_eng"] = round(eng_ratio(r["text_ko"]), 2)
    for x in f: flags[x] = flags.get(x, 0) + 1

bad = [r for r in rows if r["_flags"]]
print(f"공격 {len(rows)}건 중 의심 {len(bad)}건")
for k, v in sorted(flags.items(), key=lambda x: -x[1]): print(f"  {k}: {v}건")
rt = [r["_ratio"] for r in rows]
print(f"길이비 중앙 {st.median(rt):.2f}")

clean = [r for r in rows if not r["_flags"]]
print(f"\n자동 통과 {len(clean)}건 / 검수 필요 {len(bad)}건")
json.dump(rows, open(HERE/"jot_2023_12_ko_flagged.json","w"), ensure_ascii=False, indent=1)

# 한국어 평가셋: 통과분 공격 + WildChat 한국어 정상
neg = [json.loads(l)["text"] for l in open(HERE/"wildchat_ko.jsonl", encoding="utf-8")]
neg = list(dict.fromkeys(neg))
import random; random.Random(0).shuffle(neg)
out = HERE/"data_koeval"; out.mkdir(exist_ok=True)
n_val_neg = 3000
va = [{"text": t, "label": 0} for t in neg[:n_val_neg]]
te = ([{"text": r["text_ko"], "label": 1, "uid": r["uid"]} for r in clean]
      + [{"text": t, "label": 0} for t in neg[n_val_neg:n_val_neg+4000]])
for nm, part in (("val", va), ("test", te)):
    (out/f"{nm}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in part),
                                   encoding="utf-8")
    print(f"  data_koeval/{nm}.jsonl {len(part)}건 (공격 {sum(x['label'] for x in part)})")

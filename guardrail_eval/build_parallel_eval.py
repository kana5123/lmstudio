"""영·한 완전 병렬 평가셋. 공격·정상 모두 같은 항목, 양쪽 다 TowerInstruct 번역.

이렇게 해야 번역 흔적(translationese)이 공격/정상을 가르는 신호가 되지 않는다.
"""
import json, glob, re
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

HERE = Path(__file__).resolve().parent
CAP = 8000
pos = {r["uid"]: r for r in json.load(open(HERE/"jot_2023_12_ko_flagged.json", encoding="utf-8"))}
neg = {}
for f in sorted(glob.glob(str(HERE/"jot_2023_12_neg_ko_*.json"))):
    for r in json.load(open(f, encoding="utf-8")): neg[r["uid"]] = r
print(f"공격 {len(pos)} · 정상 {len(neg)}")

OTHER = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ぀-ヿ一-鿿]")
HAN = re.compile(r"[가-힣]")
def ok(r):
    t = r["text_ko"]
    if not HAN.search(t): return False
    ratio = len(t)/max(len(r["prompt"]),1)
    return 0.15 < ratio < 2.5 and not OTHER.search(t)
pos_ok = {u:r for u,r in pos.items() if len(r["prompt"])<=CAP and ok(r)}
neg_ok = {u:r for u,r in neg.items() if len(r["prompt"])<=CAP and ok(r)}
print(f"  자동검사 통과: 공격 {len(pos_ok)} / 정상 {len(neg_ok)}")

import random
rng = random.Random(0)
neg_ids = sorted(neg_ok); rng.shuffle(neg_ids)
pos_ids = sorted(pos_ok); rng.shuffle(pos_ids)
# 검증(임계용)에는 정상만 -- pick_threshold 는 음성만 씀
n_val = 400
val_ids = neg_ids[:n_val]; test_neg = neg_ids[n_val:]
out = HERE/"data_par"; out.mkdir(exist_ok=True)
for lang, key in (("en","prompt"), ("ko","text_ko")):
    src = {**pos_ok, **neg_ok}
    va = [{"text": src[u][key], "label": 0, "uid": u} for u in val_ids]
    te = ([{"text": pos_ok[u][key], "label": 1, "uid": u} for u in pos_ids]
          + [{"text": neg_ok[u][key], "label": 0, "uid": u} for u in test_neg])
    for nm, part in (("val", va), ("test", te)):
        (out/f"{lang}_{nm}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False)+"\n" for x in part), encoding="utf-8")
    print(f"  data_par/{lang}_test.jsonl {len(te)}건 (공격 {sum(x['label'] for x in te)})")
# 항목 일치 확인
a=[json.loads(l)["uid"] for l in open(out/"en_test.jsonl",encoding="utf-8")]
b=[json.loads(l)["uid"] for l in open(out/"ko_test.jsonl",encoding="utf-8")]
print(f"  영·한 항목 순서까지 동일: {a==b}")

"""영어 전용 학습셋(WildGuardTrain) + 한국어 평가셋(PolyGuardPrompts) 준비.

한국어는 학습에 한 건도 들어가지 않는다.
PolyGuardPrompts 는 17개 언어가 같은 1,725개 항목의 번역이므로,
영어/한국어에 '같은 인덱스'로 val/test 를 나눠 병렬성을 유지한다.
"""
import json, random
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
SEED = 0

# ── 영어 학습셋 ──
wg = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
rows = [{"text": r["prompt"], "label": int(r["prompt_harm_label"] == "harmful")}
        for r in wg if r["prompt_harm_label"] in ("harmful", "unharmful") and r["prompt"]]
rows = list({r["text"]: r for r in rows}.values())          # 중복 제거
random.Random(SEED).shuffle(rows)
d = HERE / "data_wg"; d.mkdir(exist_ok=True)
n_val = 3000
for nm, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
    (d / f"{nm}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
    print(f"data_wg/{nm}.jsonl  {len(part)}건  유해 {sum(r['label'] for r in part)}")

# ── 한국어·영어 평가셋 (같은 항목, 같은 분할) ──
pg = load_dataset("ToxicityPrompts/PolyGuardPrompts")["test"]
by = {}
for r in pg:
    if r["prompt_harm_label"] not in ("harmful", "unharmful"): continue
    by.setdefault(r["language"], {})[r["id"]] = r
ids = sorted(set(by["English"]) & set(by["Korean"]))        # 양쪽 다 있는 항목만
random.Random(SEED).shuffle(ids)
k = int(len(ids) * 0.4)
splits = {"val": ids[:k], "test": ids[k:]}
print(f"\n영·한 공통 항목 {len(ids)}건 → 검증 {k} / 시험 {len(ids)-k}")

e = HERE / "data_pgp"; e.mkdir(exist_ok=True)
for lang, tag in (("English", "en"), ("Korean", "ko")):
    for sp, sel in splits.items():
        part = [{"text": by[lang][i]["prompt"],
                 "label": int(by[lang][i]["prompt_harm_label"] == "harmful"),
                 "adversarial": bool(by[lang][i]["adversarial"]), "id": i} for i in sel]
        (e / f"{tag}_{sp}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        print(f"data_pgp/{tag}_{sp}.jsonl  {len(part)}건  유해 {sum(r['label'] for r in part)}")

# ── 누수 검사: 영어 학습셋 ∩ 평가셋 ──
tr = {r["text"] for r in rows}
for tag in ("en", "ko"):
    for sp in ("val", "test"):
        ev = {json.loads(l)["text"] for l in open(e / f"{tag}_{sp}.jsonl", encoding="utf-8")}
        n = len(tr & ev)
        print(f"누수 WildGuardTrain ∩ {tag}_{sp} = {n}건")
        assert n == 0 or tag == "en", f"{tag}_{sp} 누수 {n}건"

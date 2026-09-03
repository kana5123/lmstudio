"""과제 = '지시/안전장치를 덮어쓰려는 시도' (Prompt Guard 2 정의) 로 데이터 재구성.

라벨은 adversarial. WildGuardMix 와 PolyGuardPrompts 가 같은 이름의 열을 갖는다.
한국어는 평가에만 쓰고 학습에는 0건 들어간다.
"""
import json, random
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

HERE = Path(__file__).resolve().parent; SEED = 0

wg = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
rows = [{"text": r["prompt"], "label": int(r["adversarial"])}
        for r in wg if r["prompt"] and r["adversarial"] is not None]
rows = list({r["text"]: r for r in rows}.values())
random.Random(SEED).shuffle(rows)
d = HERE / "data_adv"; d.mkdir(exist_ok=True)
n_val = 3000
for nm, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
    (d/f"{nm}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in part),
                                 encoding="utf-8")
    print(f"data_adv/{nm}.jsonl  {len(part)}건  적대적 {sum(r['label'] for r in part)}")

# 평가셋은 data_pgp 재사용 -- label 열만 adversarial 로 바꿔 저장
e = HERE / "data_pgp"
for tag in ("en", "ko"):
    for sp in ("val", "test"):
        src = [json.loads(l) for l in open(e/f"{tag}_{sp}.jsonl", encoding="utf-8")]
        out = [{"text": r["text"], "label": int(r["adversarial"]), "id": r["id"]} for r in src]
        (e/f"adv_{tag}_{sp}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in out), encoding="utf-8")
        print(f"data_pgp/adv_{tag}_{sp}.jsonl  {len(out)}건  적대적 {sum(r['label'] for r in out)}")

tr = {r["text"] for r in rows}
for tag in ("en", "ko"):
    for sp in ("val", "test"):
        ev = {json.loads(l)["text"] for l in open(e/f"adv_{tag}_{sp}.jsonl", encoding="utf-8")}
        print(f"누수 학습 ∩ adv_{tag}_{sp} = {len(tr & ev)}건")

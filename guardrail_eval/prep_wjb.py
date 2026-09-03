"""WildJailbreak (AI2, NeurIPS 2024) -> 영어 학습셋. 한국어는 0건.

라벨 1 = 유해 요청 (vanilla_harmful + adversarial_harmful)
라벨 0 = 정상      (vanilla_benign  + adversarial_benign)
본문은 adversarial 이 있으면 그것을, 없으면 vanilla 를 쓴다.
"""
import json, random
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

HERE = Path(__file__).resolve().parent; SEED = 0; N_PER = 50000
d = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
rows = []
for r in d:
    txt = (r["adversarial"] or "").strip() or (r["vanilla"] or "").strip()
    if not txt: continue
    rows.append({"text": txt, "label": int("harmful" in r["data_type"]),
                 "adv": int(r["data_type"].startswith("adversarial"))})
rows = list({r["text"]: r for r in rows}.values())
rng = random.Random(SEED)
pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
rng.shuffle(pos); rng.shuffle(neg)
sub = pos[:N_PER] + neg[:N_PER]; rng.shuffle(sub)
print(f"중복 제거 {len(rows)} → 표본 {len(sub)} (유해 {sum(r['label'] for r in sub)}, "
      f"제일브레이크풍 {sum(r['adv'] for r in sub)})")

o = HERE/"data_wjb"; o.mkdir(exist_ok=True)
n_val, n_test = 3000, 3000
for nm, part in (("val", sub[:n_val]), ("test", sub[n_val:n_val+n_test]),
                 ("train", sub[n_val+n_test:])):
    (o/f"{nm}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in part),
                                 encoding="utf-8")
    print(f"  data_wjb/{nm}.jsonl {len(part)}건  유해 {sum(r['label'] for r in part)}")

# 한국어 평가셋과의 누수 확인
import csv, glob
mj = glob.glob("/home/kana5123/.cache/huggingface/hub/datasets--DAMO-NLP-SG--MultiJail/snapshots/*/MultiJail.csv")[0]
mjen = {r["en"].strip() for r in csv.DictReader(open(mj, encoding="utf-8"))}
tr = {r["text"] for r in sub}
print(f"누수 학습 ∩ MultiJail(영어원문) = {len(tr & mjen)}건")

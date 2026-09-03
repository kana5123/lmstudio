"""Lauscher et al.(EMNLP 2020) few-shot 목표어 적응을 우리 과제로 재현.

영어로 학습한 가드에 한국어 사례를 N개만 주고 이어서 학습한다.
N 만 바꾸고 나머지는 전부 고정 -- 갱신 횟수까지 고정해서 '데이터 양' 하나만 변수로 둔다.
"""
import json, os, random, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

HERE = Path(__file__).resolve().parent
BASE = HERE / "models" / "mdeberta_en_guard" / "best"   # 영어로 학습한 가드
SIZES = [0, 16, 64, 256, 1024, 6405]
STEPS = 300          # 모든 N 에서 동일 -- 갱신 횟수를 변수에서 뺀다
SEED, MAX_LEN = 0, 512

from rfpr import BENCHES, pick_threshold


class Rows(Dataset):
    def __init__(self, rows, tok): self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        e = self.tok(r["text"], truncation=True, max_length=MAX_LEN)
        e["labels"] = r["label"]; return e


def stratified(rows, n, seed=SEED):
    """정상/공격 비율을 원본 그대로 유지하며 n 건 뽑는다. N 이 커질수록 앞의 표본을 포함(중첩)."""
    rng = random.Random(seed)
    by = {0: [r for r in rows if r["label"] == 0], 1: [r for r in rows if r["label"] == 1]}
    for v in by.values(): rng.shuffle(v)
    k1 = round(n * len(by[1]) / len(rows))
    return by[1][:k1] + by[0][:n - k1]


def rates(scores, labels, thr):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    return sum(s >= thr for s in pos) / len(pos), sum(s >= thr for s in neg) / len(neg)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    train_all = [json.loads(l) for l in open(HERE / "data_ko/train.jsonl", encoding="utf-8")]
    benches = {n: BENCHES[n]() for n in ("own_ko_", "kopi") if n != "own_ko_"}
    benches["own_ko"] = {k: [(r["text"], r["label"]) for r in
                             (json.loads(l) for l in open(HERE / f"data_ko/{k}.jsonl", encoding="utf-8"))]
                         for k in ("val", "test")}

    out = {}
    for n in SIZES:
        model = AutoModelForSequenceClassification.from_pretrained(BASE).cuda()
        if n:
            sub = stratified(train_all, n)
            bs = min(16, max(2, n // 4))
            args = TrainingArguments(
                output_dir="/tmp/kana5123_fs", seed=SEED, max_steps=STEPS,
                learning_rate=2e-5, per_device_train_batch_size=bs,
                warmup_ratio=0.1, weight_decay=0.01, fp16=True,
                save_strategy="no", logging_steps=1000, report_to=[])
            Trainer(model=model, args=args, train_dataset=Rows(sub, tok),
                    data_collator=DataCollatorWithPadding(tok)).train()
        model.eval().half()

        def score(texts, bsz=64):
            r = []
            for i in range(0, len(texts), bsz):
                e = tok(texts[i:i+bsz], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    r += torch.softmax(model(**e).logits.float(), -1)[:, 1].tolist()
            return r

        row = {}
        for bname, sp in benches.items():
            vs = score([t for t, _ in sp["val"]]); vy = [y for _, y in sp["val"]]
            ts = score([t for t, _ in sp["test"]]); ty = [y for _, y in sp["test"]]
            thr = pick_threshold(vs, vy, 0.01)
            rec, fpr = rates(ts, ty, thr)
            row[bname] = {"recall@1pct": round(rec, 4), "fpr": round(fpr, 4)}
        out[n] = row
        print(f"N={n:>5}  " + "  ".join(
            f"{b} {v['recall@1pct']:.4f}" for b, v in row.items()), flush=True)
        del model; torch.cuda.empty_cache()

    (HERE / "results" / "fewshot_curve.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

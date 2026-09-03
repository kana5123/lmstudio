"""mDeBERTa-v3-base 를 한국어 프롬프트 인젝션 이진 분류로 파인튜닝.

라벨 0 = 정상(safe), 1 = 인젝션 공격(unsafe).
평가는 이 프로젝트의 가드레일 비교와 같은 축을 쓴다:
  일반 Recall(기본 임계 0.5) / Recall@1%FPR / Recall@0.1%FPR.
FPR 고정 임계값은 검증셋에서만 고르고 시험셋에 그대로 적용한다.
"""
import json, os, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

HERE = Path(__file__).resolve().parent
DATA = HERE / os.environ.get("DATA_DIR", "data_ko")
OUTDIR = HERE / "models" / os.environ.get("RUN", "mdeberta_ko_guard")
# BASE 를 주면 그 체크포인트에서 이어서 학습한다 (영어 학습 -> 한국어 전이).
MODEL = os.environ.get("BASE", "microsoft/mdeberta-v3-base")
if (HERE / MODEL).exists():          # 로컬 체크포인트면 절대경로로, 아니면 허브 id 그대로
    MODEL = str(HERE / MODEL)
MAX_LEN = 512
SEED = 0


class Jsonl(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = r["label"]
        return enc


def pick_threshold(scores, labels, target_fpr):
    """검증셋에서 FPR <= target 을 만족하는 가장 낮은 임계값."""
    neg = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    if not neg:
        return 1.0
    k = int(len(neg) * target_fpr)
    return float(neg[k]) + 1e-12 if k < len(neg) else 0.0


def rates(scores, labels, thr):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    rec = sum(s >= thr for s in pos) / len(pos) if pos else float("nan")
    fpr = sum(s >= thr for s in neg) / len(neg) if neg else float("nan")
    return rec, fpr


def scores_of(trainer, ds):
    logits = trainer.predict(ds).predictions
    p = torch.softmax(torch.tensor(logits).float(), dim=-1)[:, 1].numpy()
    return p, np.array([r["label"] for r in ds.rows])


def main():
    epochs = float(sys.argv[1]) if len(sys.argv) > 1 else 3
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Prompt Guard 는 라벨이 3개(BENIGN/INJECTION/JAILBREAK)라 헤드만 새로 만든다.
    # 인코더 가중치는 그대로 이어받는다 -- 백본 구조가 mDeBERTa-v3-base 와 동일.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, ignore_mismatched_sizes=True)
    tr, va, te = (Jsonl(DATA / f"{n}.jsonl", tok) for n in ("train", "val", "test"))
    print(f"학습 {len(tr)} / 검증 {len(va)} / 시험 {len(te)}")

    args = TrainingArguments(
        output_dir=str(OUTDIR), seed=SEED,
        num_train_epochs=epochs, learning_rate=2e-5,
        per_device_train_batch_size=int(os.environ.get("BS","16")), per_device_eval_batch_size=64,
        warmup_ratio=0.1, weight_decay=0.01, fp16=True,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        logging_steps=50, report_to=[])
    from transformers import DataCollatorWithPadding
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    print(f"학습 {time.time()-t0:.0f}초")

    vs, vy = scores_of(trainer, va)
    ts, ty = scores_of(trainer, te)
    res = {}
    rec, fpr = rates(ts, ty, 0.5)
    res["기본(0.5)"] = dict(recall=rec, fpr=fpr)
    for tgt, key in ((0.01, "1pct"), (0.001, "0.1pct")):
        thr = pick_threshold(vs, vy, tgt)
        rec, fpr = rates(ts, ty, thr)
        res[f"FPR {tgt*100:g}%"] = dict(thr=thr, recall=rec, fpr=fpr)
    print("\n=== 시험셋 결과")
    for k, v in res.items():
        extra = f"  임계 {v['thr']:.6f}" if "thr" in v else ""
        print(f"  {k:10} Recall {v['recall']:.4f}  달성 FPR {v['fpr']*100:.2f}%{extra}")
    (OUTDIR / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    trainer.save_model(str(OUTDIR / "best"))
    tok.save_pretrained(str(OUTDIR / "best"))


if __name__ == "__main__":
    main()

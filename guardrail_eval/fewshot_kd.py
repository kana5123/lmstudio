"""교차언어 증류 + 한국어 소량 추가 (few-shot 목표어 적응).

  근거 ① Xu & Yang, ACL 2017 — 교사가 '영어 원문'에 매긴 확률을 그 번역문에 붙여 학습.
          한국어 라벨은 여전히 0건이다.
  근거 ② Lauscher et al., EMNLP 2020 — 소량 목표어 사례가 zero-shot 격차를 메운다.

N(한국어 추가량)만 바꾸고 나머지는 전부 고정한다.
"""
import json, os, sys, time
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

HERE = Path(__file__).resolve().parent
TEACHER = "meta-llama/Llama-Prompt-Guard-2-86M"
STUDENT = "microsoft/mdeberta-v3-base"
N_KO   = int(os.environ.get("N_KO", "0"))
OUT    = HERE / "models" / os.environ.get("RUN", f"fs_ko{N_KO}")
MAX_LEN, SEED = 512, 0
BS = int(os.environ.get("BS", "32"))


def teacher_scores(tok, texts, bs=128):
    """교사는 영어만 본다. 학습하지 않는다."""
    m = AutoModelForSequenceClassification.from_pretrained(TEACHER).cuda().eval().half()
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out += torch.softmax(m(**e).logits.float(), -1).cpu().tolist()
    del m; torch.cuda.empty_cache()
    return out


class KDSet(Dataset):
    def __init__(self, texts, soft, tok): self.t, self.q, self.tok = texts, soft, tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=MAX_LEN)
        e["soft"] = self.q[i]; return e


class KDCollator(DataCollatorWithPadding):
    def __call__(self, feats):
        soft = torch.tensor([f.pop("soft") for f in feats])
        b = super().__call__(feats); b["soft"] = soft; return b


class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        soft = inputs.pop("soft"); inputs.pop("labels", None)
        logits = model(**inputs).logits
        loss = -(soft * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
        return (loss, logits) if return_outputs else loss


def main():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    en = [json.loads(l)["text"] for l in open(HERE/"data_jot/train.jsonl", encoding="utf-8")]
    print(f"영어 {len(en)}건 · 한국어 추가 {N_KO}건", flush=True)
    t0 = time.time()
    q_en = teacher_scores(tok, en)
    texts, soft = list(en), list(q_en)

    if N_KO:
        pairs = json.load(open(HERE/"jot_train_ko.json", encoding="utf-8"))[:N_KO]
        # 교사는 '영어 원문'을 채점하고, 그 확률을 한국어 번역문에 붙인다
        q_ko = teacher_scores(tok, [p["prompt"] for p in pairs])
        texts += [p["text_ko"] for p in pairs]; soft += q_ko
        print(f"  한국어 {len(pairs)}건 추가 (라벨은 영어 원문에 대한 교사 확률)", flush=True)
    print(f"  총 {len(texts)}건, 교사 채점 {time.time()-t0:.0f}초", flush=True)

    args = TrainingArguments(
        output_dir=str(OUT), seed=SEED, num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=BS, warmup_ratio=0.1, weight_decay=0.01, fp16=True,
        eval_strategy="no", save_strategy="no", logging_steps=300, report_to=[],
        remove_unused_columns=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=2, ignore_mismatched_sizes=True)
    KDTrainer(model=model, args=args, train_dataset=KDSet(texts, soft, tok),
              data_collator=KDCollator(tok)).train()
    model.save_pretrained(str(OUT/"best")); tok.save_pretrained(str(OUT/"best"))
    print(f"저장 → {OUT/'best'}  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()

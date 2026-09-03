"""MONOX-KD (Chi et al., AACL 2020) — 교사의 soft label 로 학생을 학습.

전부 영어에서 일어난다. 한국어는 평가에만 등장한다.
손실  L_KD = − Σ_x Σ_k q(y=k|x; 교사) · log p(y=k|x; 학생)      (교사 고정)
"""
import json, os, time
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

HERE = Path(__file__).resolve().parent
TEACHER = HERE / "models" / os.environ.get("TEACHER", "adv_teacher_pg2") / "best"
STUDENT = os.environ.get("STUDENT", "microsoft/mdeberta-v3-base")
OUTDIR  = HERE / "models" / os.environ.get("RUN", "adv_student_monox")
DATA    = HERE / os.environ.get("DATA_DIR", "data_adv")
TEMP    = float(os.environ.get("TEMP", "1.0"))     # 논문은 0.1 도 실험
MAX_LEN, SEED = 512, 0


def soft_labels(tok, rows, bs=128):
    """교사를 영어 텍스트에 돌려 확률분포를 받아둔다. 교사는 갱신되지 않는다."""
    m = AutoModelForSequenceClassification.from_pretrained(TEACHER).cuda().eval().half()
    out = []
    t0 = time.time()
    for i in range(0, len(rows), bs):
        e = tok([r["text"] for r in rows[i:i+bs]], truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out += torch.softmax(m(**e).logits.float() / TEMP, -1).cpu().tolist()
    print(f"  교사 채점 {len(out)}건 {time.time()-t0:.0f}초")
    del m; torch.cuda.empty_cache()
    return out


class KDSet(Dataset):
    def __init__(self, rows, q, tok): self.rows, self.q, self.tok = rows, q, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        e = self.tok(self.rows[i]["text"], truncation=True, max_length=MAX_LEN)
        e["labels"] = self.rows[i]["label"]          # 참고용(손실에 안 씀)
        e["soft"] = self.q[i]
        return e


class KDCollator(DataCollatorWithPadding):
    def __call__(self, feats):
        soft = torch.tensor([f.pop("soft") for f in feats])
        b = super().__call__(feats); b["soft"] = soft
        return b


class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        soft = inputs.pop("soft"); inputs.pop("labels", None)
        logits = model(**inputs).logits
        loss = -(soft * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
        return (loss, logits) if return_outputs else loss


def main():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    J = lambda n: [json.loads(l) for l in open(DATA/f"{n}.jsonl", encoding="utf-8")]
    tr, va = J("train"), J("val")
    print(f"학습 {len(tr)} / 검증 {len(va)}  (전부 영어)")
    q_tr = soft_labels(tok, tr)

    args = TrainingArguments(
        output_dir=str(OUTDIR), seed=SEED, num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=int(os.environ.get("BS", "32")),
        warmup_ratio=0.1, weight_decay=0.01, fp16=True,
        eval_strategy="no", save_strategy="epoch", save_total_limit=1,
        logging_steps=200, report_to=[], remove_unused_columns=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=2, ignore_mismatched_sizes=True)
    t0 = time.time()
    KDTrainer(model=model, args=args, train_dataset=KDSet(tr, q_tr, tok),
              data_collator=KDCollator(tok)).train()
    print(f"증류 {time.time()-t0:.0f}초")
    model.save_pretrained(str(OUTDIR/"best")); tok.save_pretrained(str(OUTDIR/"best"))
    print(f"저장 → {OUTDIR/'best'}")


if __name__ == "__main__":
    main()

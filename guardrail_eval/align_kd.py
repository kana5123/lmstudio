"""과제 증류 + 중간층 표현 정렬 (Liu & Niehues, ACL 2025 방식을 인코더에 적용).

  L = L_과제(영어, 교사 확률)  +  λ · L_정렬(영·한 병렬, i층)

  L_정렬 = −log  exp(sim(h_s^i, h_t^i)/τ) / Σ_v exp(sim(h_s^i, h_v^i)/τ)
           같은 묶음 안에서 '진짜 번역 짝' 만 가깝게, 나머지는 멀게.
           ★ 과제 라벨 불필요 — 번역 짝이라는 정보만 씀.

LAYER=0 이면 정렬 손실 없음(기준선 A).
"""
import json, os, time
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

HERE   = Path(__file__).resolve().parent
TEACHER= "meta-llama/Llama-Prompt-Guard-2-86M"
STUDENT= "microsoft/mdeberta-v3-base"
LAYER  = int(os.environ.get("LAYER", "8"))      # 0 = 정렬 없음
LAM    = float(os.environ.get("LAM", "1.0"))
TAU    = float(os.environ.get("TAU", "0.05"))
KO_TASK= os.environ.get("KO_TASK", "no")   # no | add(영+한) | only(한국어만)
OUT    = HERE/"models"/os.environ.get("RUN", f"align_L{LAYER}")
BS     = int(os.environ.get("BS", "16"))
MAX_LEN, SEED = 512, 0


def teacher_scores(tok, texts, bs=128):
    m = AutoModelForSequenceClassification.from_pretrained(TEACHER).cuda().eval().half()
    o=[]
    for i in range(0,len(texts),bs):
        e=tok(texts[i:i+bs],truncation=True,max_length=MAX_LEN,padding=True,
              return_tensors="pt").to("cuda")
        with torch.no_grad(): o+=torch.softmax(m(**e).logits.float(),-1).cpu().tolist()
    del m; torch.cuda.empty_cache(); return o


class DS(Dataset):
    """영어 문장 + 교사 확률. 정렬용으로 (영어, 한국어) 짝을 돌아가며 하나씩 붙인다."""
    def __init__(self, en, soft, pairs, tok):
        self.en, self.soft, self.pairs, self.tok = en, soft, pairs, tok
    def __len__(self): return len(self.en)
    def __getitem__(self, i):
        e = self.tok(self.en[i], truncation=True, max_length=MAX_LEN)
        e["soft"] = self.soft[i]
        if self.pairs:
            p = self.pairs[i % len(self.pairs)]
            e["_a_en"] = self.tok(p["prompt"],   truncation=True, max_length=MAX_LEN)["input_ids"]
            e["_a_ko"] = self.tok(p["text_ko"], truncation=True, max_length=MAX_LEN)["input_ids"]
        return e


class Coll(DataCollatorWithPadding):
    def __call__(self, feats):
        soft = torch.tensor([f.pop("soft") for f in feats])
        aen  = [f.pop("_a_en") for f in feats] if "_a_en" in feats[0] else None
        ako  = [f.pop("_a_ko") for f in feats] if "_a_ko" in feats[0] else None
        b = super().__call__(feats); b["soft"] = soft
        if aen is not None:
            for k, seqs in (("ae", aen), ("ak", ako)):
                p = self.tokenizer.pad({"input_ids": seqs}, return_tensors="pt")
                b[k+"_ids"], b[k+"_mask"] = p["input_ids"], p["attention_mask"]
        return b


def pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h*m).sum(1)/m.sum(1)


class AlignTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        soft = inputs.pop("soft"); inputs.pop("labels", None)
        ae = {k[3:]: inputs.pop(k) for k in list(inputs) if k.startswith("ae_")}
        ak = {k[3:]: inputs.pop(k) for k in list(inputs) if k.startswith("ak_")}
        out = model(**inputs)
        L_task = -(soft * torch.log_softmax(out.logits.float(), -1)).sum(-1).mean()
        if LAYER == 0 or not ae:
            return (L_task, out.logits) if return_outputs else L_task
        hs_en = model(input_ids=ae["ids"], attention_mask=ae["mask"],
                      output_hidden_states=True).hidden_states[LAYER]
        hs_ko = model(input_ids=ak["ids"], attention_mask=ak["mask"],
                      output_hidden_states=True).hidden_states[LAYER]
        a = F.normalize(pool(hs_en, ae["mask"]).float(), dim=-1)
        b = F.normalize(pool(hs_ko, ak["mask"]).float(), dim=-1)
        sim = (a @ b.T) / TAU                        # 대각선이 진짜 번역 짝
        L_align = F.cross_entropy(sim, torch.arange(len(a), device=sim.device))
        loss = L_task + LAM * L_align
        return (loss, out.logits) if return_outputs else loss


def main():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    en = [json.loads(l)["text"] for l in open(HERE/"data_jot/train.jsonl", encoding="utf-8")]
    allpairs = json.load(open(HERE/"jot_train_ko.json", encoding="utf-8"))
    pairs = allpairs if LAYER else []
    t0=time.time()
    if KO_TASK == "only":
        # 한국어만 과제 학습 (교사는 영어 원문을 채점)
        task_txt = [p["text_ko"] for p in allpairs]
        soft = teacher_scores(tok, [p["prompt"] for p in allpairs])
    else:
        task_txt, soft = list(en), teacher_scores(tok, en)
        if KO_TASK == "add":
            # translate-train: 한국어 번역문에 '영어 원문에 대한 교사 확률' 을 붙임
            task_txt += [p["text_ko"] for p in allpairs]
            soft += teacher_scores(tok, [p["prompt"] for p in allpairs])
    print(f"정렬 층 {LAYER} · λ {LAM} · KO_TASK={KO_TASK} · "
          f"과제 {len(task_txt)}건 · 정렬 병렬 {len(pairs)}쌍", flush=True)
    print(f"  교사 채점 {time.time()-t0:.0f}초", flush=True)
    args = TrainingArguments(output_dir=str(OUT), seed=SEED, num_train_epochs=3,
        learning_rate=2e-5, per_device_train_batch_size=BS, warmup_ratio=0.1,
        weight_decay=0.01, fp16=True, eval_strategy="no", save_strategy="no",
        logging_steps=400, report_to=[], remove_unused_columns=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=2, ignore_mismatched_sizes=True)
    AlignTrainer(model=model, args=args, train_dataset=DS(task_txt, soft, pairs, tok),
                 data_collator=Coll(tok)).train()
    model.save_pretrained(str(OUT/"best")); tok.save_pretrained(str(OUT/"best"))
    print(f"저장 → {OUT/'best'}  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()

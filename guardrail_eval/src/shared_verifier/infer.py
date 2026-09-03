"""PHASE1-3: 호환 (모델 x 데이터셋) 쌍 전부에 대해 추론 + 층별 CLS 은닉표현 저장.

모델은 전부 동결이다.  어떤 모델 파일도 수정하지 않는다.
PIGuard 만 forward 가 output_hidden_states=False 로 하드코딩돼 있어
내부 서브모듈(m.deberta)을 직접 호출한다 -- 로짓이 원래 forward 와
일치하는지 실행 시 assert 로 확인한다.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared_verifier.registry import MODELS, attack_prob

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/shared_verifier/hidden"
MAXLEN, BS = 512, 64


def resolve(df, key):
    """호환표의 데이터셋 키 -> canonical_samples 부분집합."""
    fam, name = key.split(":", 1)
    if fam == "piguard":
        return df[(df.canonical_dataset == "piguard_train_mix") & (df.original_source == name)]
    if fam == "promptshield":
        return df[df.original_source == f"promptshield_{name}"]
    if fam == "wildjailbreak":
        return df[(df.canonical_dataset == "wildjailbreak")
                  & (df.original_source.str.startswith(name))]
    raise KeyError(key)


@torch.no_grad()
def run_model(mkey, tok, model, texts, dev):
    """길이순 정렬로 배치 -> (n, L+1, 768) fp16 CLS, (n, C) 로짓."""
    order = np.argsort([-len(t) for t in texts])
    H, LG = [None] * len(texts), [None] * len(texts)
    custom = MODELS[mkey].get("custom", False)
    for s in range(0, len(order), BS):
        idx = order[s:s + BS]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(dev)
        if custom:
            # PIGuard.forward 가 은닉표현을 막아둬서 내부 인코더를 직접 부른다.
            core = model.deberta(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"],
                                 output_hidden_states=True)
            logits = model.classifier(core.last_hidden_state[:, 0, :])
            hs = core.hidden_states
            ref = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            assert torch.allclose(logits, ref, atol=1e-5), "PIGuard 우회 경로가 원래 로짓과 다름"
        else:
            o = model(**enc, output_hidden_states=True)
            logits, hs = o.logits, o.hidden_states
        cls = torch.stack([h[:, 0, :] for h in hs], 1).half().cpu()   # (b, L+1, 768)
        for j, i in enumerate(idx):
            H[i], LG[i] = cls[j], logits[j].float().cpu()
    return torch.stack(H), torch.stack(LG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="쉼표 구분, 비우면 전체")
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data/multisource_guard/canonical_samples.parquet")
    pairs = [r for r in csv.DictReader(open(ROOT / "results/shared_verifier/task_compatibility.csv"))
             if r["compatible"] in ("yes", "partial")]
    want = set(args.models.split(",")) if args.models else set(MODELS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    for mkey in [m for m in MODELS if m in want]:
        todo = [p for p in pairs if p["model_key"] == mkey
                and not (OUT / f"{mkey}__{p['dataset'].replace(':', '-')}.pt").exists()]
        if not todo:
            print(f"[{mkey}] 이미 완료", flush=True)
            continue
        hf = MODELS[mkey]["hf"]
        tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf, trust_remote_code=True).eval().to(dev)
        for p in todo:
            sub = resolve(df, p["dataset"])
            # 정답 라벨이 없는 행(binary_main_eligible=False)은 혼동 셀을 만들 수 없다
            drop = (~sub.binary_main_eligible).sum()
            sub = sub[sub.binary_main_eligible].reset_index(drop=True)
            h, lg = run_model(mkey, tok, model, sub.text.tolist(), dev)
            prob = torch.softmax(lg, -1).numpy()
            pa = attack_prob(mkey, prob, p["dataset_attack_family"])
            pred = (pa >= 0.5).astype(np.int8)
            y = (sub.binary_main_label == "UNSAFE").to_numpy().astype(np.int8)
            assert sub.binary_main_label.isin(["SAFE", "UNSAFE"]).all()
            cell = np.where(pred == 1, np.where(y == 1, "TP", "FP"),
                            np.where(y == 1, "FN", "TN"))
            torch.save(dict(
                model=mkey, hf=hf, dataset=p["dataset"], layers=MODELS[mkey]["layers"],
                attack_family=p["dataset_attack_family"], compatible=p["compatible"],
                sample_id=sub.sample_id.tolist(), h_cls=h, logits=lg,
                p_attack=torch.from_numpy(pa.astype(np.float32)),
                pred=torch.from_numpy(pred), y=torch.from_numpy(y), cell=cell.tolist(),
                dup=sub.duplicate_group_id.tolist(), original_source=sub.original_source.tolist(),
            ), OUT / f"{mkey}__{p['dataset'].replace(':', '-')}.pt")
            u, c = np.unique(cell, return_counts=True)
            print(f"[{mkey}] {p['dataset']:<38} n={len(sub):>6}  "
                  f"{dict(zip(u, c.tolist()))}  h={tuple(h.shape)}"
                  + (f"  (라벨없음 {drop} 제외)" if drop else ""), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

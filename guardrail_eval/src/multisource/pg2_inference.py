"""동결 PromptGuard2 로 hard prediction 만 만든다 (지시문 9·10절).

채점 규칙은 검증된 JailbreaksOverTime 재현과 **완전히 동일**하게 유지한다:
  max_length=512, stride=128, 겹치는 창, 창별 UNSAFE 확률의 최댓값, 임계 0.5(=argmax).
라벨 방향도 그 실험에서 실측 확인한 값을 그대로 쓴다: LABEL_1 = MALICIOUS(UNSAFE).
(id2label 을 추측하지 않고 config 값을 assert 로 확인한다.)

확률/로짓은 metadata 로만 저장한다.  데이터셋 선정이나 향후 특징으로 쓰지 않는다.
"""
import argparse, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/multisource_guard"
ART = ROOT / "artifacts/multisource_guard"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MAX_LEN, STRIDE = 512, 128
UNSAFE_ID, BENIGN_ID = 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    assert m.config.id2label == {0: "LABEL_0", 1: "LABEL_1"}, m.config.id2label
    assert m.config.num_hidden_layers == 12 and m.config.hidden_size == 768

    df = pd.read_parquet(DATA / "canonical_samples.parquet")
    df = df[df["binary_main_eligible"]].reset_index(drop=True)
    mine = df.iloc[a.shard::a.nshards].reset_index(drop=True)
    print(f"shard {a.shard}/{a.nshards}: {len(mine)}/{len(df)}건 (MAIN 대상만)", flush=True)

    # --- 표본별 창을 평평하게 펼쳐 한 번에 배치 추론 ---
    texts = mine["text"].astype(str).tolist()
    owner, wins = [], []
    for i, t in enumerate(texts):
        e = tok(t, truncation=True, max_length=MAX_LEN, stride=STRIDE,
                return_overflowing_tokens=True)
        for ids in e["input_ids"]:
            wins.append(ids); owner.append(i)
    owner = np.array(owner)
    print(f"  창 {len(wins)}개 (표본당 평균 {len(wins)/max(len(texts),1):.2f})", flush=True)

    best = np.full(len(texts), -1.0)
    best_lo = np.zeros((len(texts), 2), dtype=np.float32)
    order = np.argsort([-len(w) for w in wins])          # 길이순 정렬로 패딩 낭비 감소
    with torch.no_grad():
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            batch = [wins[j] for j in idx]
            L = max(len(b) for b in batch)
            ii = torch.zeros(len(batch), L, dtype=torch.long)
            mm = torch.zeros(len(batch), L, dtype=torch.long)
            for r, b in enumerate(batch):
                ii[r, :len(b)] = torch.tensor(b); mm[r, :len(b)] = 1
            lo = m(input_ids=ii.to(dev), attention_mask=mm.to(dev)).logits.float().cpu()
            pr = lo.softmax(-1)[:, UNSAFE_ID].numpy()
            for r, j in enumerate(idx):
                o = owner[j]
                if pr[r] > best[o]:
                    best[o] = pr[r]; best_lo[o] = lo[r].numpy()
            if (s // a.batch) % 200 == 0:
                print(f"  {s}/{len(order)} 창", flush=True)

    out = mine[["sample_id", "source_group", "canonical_dataset", "canonical_label",
                "binary_main_label", "duplicate_group_id", "original_source",
                "attack_family", "language"]].copy()
    out["pg2_prediction"] = np.where(best >= 0.5, "UNSAFE", "SAFE")
    # 아래 세 값은 metadata 전용 — 선정/특징으로 쓰지 않는다
    out["meta_unsafe_probability"] = best
    out["meta_logit_unsafe"] = best_lo[:, UNSAFE_ID]
    out["meta_logit_benign"] = best_lo[:, BENIGN_ID]
    p = ART / f"pg2_pred_{a.shard}of{a.nshards}.parquet"
    out.to_parquet(p, index=False)
    print(f"저장 {p}  UNSAFE 예측 {(out['pg2_prediction']=='UNSAFE').sum()}/{len(out)}")


if __name__ == "__main__":
    main()

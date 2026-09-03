"""CORE 출처의 층별 CLS 은닉표현 추출 (다음 단계 H).

**일반 PromptGuard2 순전파만** 쓴다. DecompX 를 쓰지 않는다.
채점 규칙은 검증된 재현과 동일: 512 토큰 창, stride 128, 창별 UNSAFE 확률 최댓값,
그 최댓값을 낸 창에서 층별 CLS 를 뽑는다.

출력: artifacts/direction_repro/hidden_{group}.pt
  h  (n, L+1, 768)  0=임베딩, 1..12=인코더 층 출력
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
OUT = ROOT / "artifacts/direction_repro"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MAX_LEN, STRIDE, UNSAFE_ID = 512, 128, 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--cells", default="TP,FP")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    assert m.config.id2label == {0: "LABEL_0", 1: "LABEL_1"}
    L = m.config.num_hidden_layers

    cc = pd.read_parquet(DATA / "confusion_cells.parquet")
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[["sample_id", "text"]]
    sm = pd.read_parquet(DATA / "split_manifest.parquet")[["sample_id", "split", "group_key"]]
    df = (cc[cc["source_group"] == a.group]
          .merge(can, on="sample_id").merge(sm, on="sample_id"))
    df = df[df["confusion_cell"].isin(a.cells.split(","))].reset_index(drop=True)
    mine = df.iloc[a.shard::a.nshards].reset_index(drop=True)
    print(f"{a.group} {a.cells} shard {a.shard}/{a.nshards}: {len(mine)}/{len(df)}건", flush=True)

    H = torch.zeros(len(mine), L + 1, 768)
    seq = np.zeros(len(mine), dtype=np.int64)
    for i, t in enumerate(mine["text"].astype(str)):
        e = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                stride=STRIDE, return_overflowing_tokens=True, padding=True)
        e.pop("overflow_to_sample_mapping", None)
        b = {k: v.to(dev) for k, v in e.items()}
        o = m(**b, output_hidden_states=True)
        w = int(o.logits.float().softmax(-1)[:, UNSAFE_ID].argmax())   # 같은 창 선택 규칙
        H[i] = torch.stack([h[w, 0] for h in o.hidden_states]).float().cpu()
        seq[i] = int(e["attention_mask"][w].sum())
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(mine)}", flush=True)

    d = {"sample_id": mine["sample_id"].tolist(), "h": H, "layers": L,
         "cell": mine["confusion_cell"].tolist(),
         "gt": torch.tensor((mine["binary_main_label"] == "UNSAFE").astype(int).values),
         "split": mine["split"].tolist(), "group_key": mine["group_key"].tolist(),
         "text_len": torch.tensor(mine["text"].astype(str).str.len().values),
         "seq_len": torch.tensor(seq)}
    p = OUT / f"hidden_{a.group.replace(':','_')}_{a.shard}of{a.nshards}.pt"
    torch.save(d, p)
    print(f"저장 {p}  TP={int(d['gt'].sum())} FP={int((1-d['gt']).sum())}")


if __name__ == "__main__":
    main()

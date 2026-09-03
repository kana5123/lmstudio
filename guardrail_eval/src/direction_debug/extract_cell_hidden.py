"""4-cell 층별 CLS 은닉표현 추출.  **일반 PG2 순전파만** (DecompX 아님)."""
import argparse, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data/direction_debug/cell_manifest.parquet"
OUT = ROOT / "artifacts/direction_debug"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MAX_LEN, STRIDE, UNSAFE_ID = 512, 128, 1
WIN_B = 16          # 한 번에 넣을 창 수 상한 (OOM 방지)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
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

    df = pd.read_parquet(IN).reset_index(drop=True)
    mine = df.iloc[a.shard::a.nshards].reset_index(drop=True)
    print(f"shard {a.shard}/{a.nshards}: {len(mine)}/{len(df)}건", flush=True)

    H = torch.zeros(len(mine), L + 1, 768)
    conf = np.zeros((len(mine), 2), dtype=np.float32)
    for i, t in enumerate(mine["text"].astype(str)):
        e = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                stride=STRIDE, return_overflowing_tokens=True, padding=True)
        e.pop("overflow_to_sample_mapping", None)
        # 창이 많은 긴 문서에서 한 번에 다 넣으면 OOM 이 난다(실측: 공유 GPU 에서 실패).
        # WIN_B 개씩 나눠 넣고 최고 UNSAFE 확률 창만 남긴다 — 결과는 동일.
        nw = e["input_ids"].shape[0]
        best_p, best_h, best_lo = -1.0, None, None
        for k0 in range(0, nw, WIN_B):
            b = {k: v[k0:k0 + WIN_B].to(dev) for k, v in e.items()}
            o = m(**b, output_hidden_states=True)
            lo = o.logits.float()
            pr = lo.softmax(-1)[:, UNSAFE_ID]
            j = int(pr.argmax())
            if float(pr[j]) > best_p:
                best_p = float(pr[j])
                best_h = torch.stack([h[j, 0] for h in o.hidden_states]).float().cpu()
                best_lo = lo[j].cpu().numpy()
            del o, lo, pr
        H[i] = best_h
        conf[i] = best_lo                      # 진단·통제 전용 (특징으로 쓰지 않음)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(mine)}", flush=True)

    d = {"h": H, "layers": L,
         "sample_id": mine["sample_id"].tolist(),
         "dataset": mine["dataset"].tolist(),
         "cell": mine["confusion_cell"].tolist(),
         "split_role": mine["split_role"].tolist(),
         "dup": mine["duplicate_group_id"].tolist(),
         "logit_benign": torch.tensor(conf[:, 0]), "logit_unsafe": torch.tensor(conf[:, 1]),
         "text_len": torch.tensor(mine["text"].astype(str).str.len().values)}
    p = OUT / f"cellhidden_{a.shard}of{a.nshards}.pt"
    torch.save(d, p)
    print(f"저장 {p}")


if __name__ == "__main__":
    main()

"""float32 추출 정밀도가 SDR 증거를 오염시키는지 float64 기준으로 확인한다.

복원 상대오차 자체가 아니라 '증거가 달라지는가' 가 판단 근거다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier.config import BASE_MODEL, RES, load_runtime
from src.decompx_verifier.decompx_audit import DCFG
from src.decompx_verifier.sdr import sdr_evidence
from src.pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

KEYS_VEC = ["g", "zD_pos", "zD_neg", "zH_pos", "zH_neg", "zE_pos", "zE_neg"]
KEYS_SCA = ["mass_pos", "mass_neg", "g_norm"]


def collect(texts, dtype, device="cuda"):
    m, tok, rc = load_runtime(device)
    m = m.to(dtype)
    dx = DecompXDebertaV2(m)
    out = []
    for t in texts:
        e = tok(t, return_tensors="pt", truncation=False).to(device)
        with torch.no_grad():
            _, _, hs, o = dx.forward(e["input_ids"], e["attention_mask"], DCFG,
                                     output_hidden_states=True)
        ev, p, _, _, _ = sdr_evidence(o.cls_encoder, hs, e["attention_mask"])
        out.append({k: v.double().cpu() for k, v in ev.items()})
    del m, dx
    torch.cuda.empty_cache()
    return out


def main(n_per_bin=4, bins=((16, 64), (64, 128), (128, 256))):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    d = pd.read_parquet(Path(__file__).resolve().parents[2]
                        / "data/multisource_guard/canonical_samples.parquet", columns=["text"])
    rng = np.random.default_rng(0)
    d = d.iloc[rng.permutation(len(d))[:4000]].reset_index(drop=True)
    d["ntok"] = [len(tok(t, truncation=False)["input_ids"]) for t in d.text]
    pick = pd.concat([d[(d.ntok > lo) & (d.ntok <= hi)].head(n_per_bin) for lo, hi in bins])

    a = collect(pick.text.tolist(), torch.float32)
    b = collect(pick.text.tolist(), torch.float64)
    rows = []
    for k in KEYS_VEC:
        cs, rl = [], []
        for x, y in zip(a, b):
            x, y = x[k].reshape(-1, x[k].shape[-1]), y[k].reshape(-1, y[k].shape[-1])
            n = y.norm(dim=-1); keep = n > 1e-9
            if keep.any():
                cs.append(float(torch.cosine_similarity(x[keep], y[keep], dim=-1).min()))
                rl.append(float(((x[keep] - y[keep]).norm(dim=-1) / n[keep]).max()))
        rows.append(dict(evidence=k, kind="vector", cos_min=min(cs), rel_l2_max=max(rl)))
    for k in KEYS_SCA:
        rl = max(float(((x[k] - y[k]).abs() / (y[k].abs() + 1e-12)).max()) for x, y in zip(a, b))
        rows.append(dict(evidence=k, kind="scalar", cos_min=np.nan, rel_l2_max=rl))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "precision_audit.csv", index=False)
    print(f"표본 {len(pick)}개 (토큰 {pick.ntok.min()}~{pick.ntok.max()})")
    print(df.round(8).to_string(index=False))
    print(f"저장 -> {RES/'precision_audit.csv'}")
    return df


if __name__ == "__main__":
    main()

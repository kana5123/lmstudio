"""PHASE A 수치 감사 실행.  길이 구간별 층화 표본으로 정확도와 비용을 함께 잰다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_verifier.config import RES
from src.decompx_verifier.decompx_audit import run, verdict
from transformers import AutoTokenizer

N_PER_BIN = 12
BINS = [(1, 32), (32, 64), (64, 128), (128, 256), (256, 512)]

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-Prompt-Guard-2-86M")
d = pd.read_parquet(Path(__file__).resolve().parents[2]
                    / "data/multisource_guard/canonical_samples.parquet",
                    columns=["sample_id", "text", "canonical_dataset"])
rng = np.random.default_rng(0)
d = d.iloc[rng.permutation(len(d))[:6000]].reset_index(drop=True)
d["ntok"] = [len(tok(t, truncation=False)["input_ids"]) for t in d.text]
d = d[d.ntok <= 512]

pick = pd.concat([d[(d.ntok > lo) & (d.ntok <= hi)].head(N_PER_BIN) for lo, hi in BINS])
print(f"감사 표본 {len(pick)}개, 길이 분포 {pick.ntok.min()}~{pick.ntok.max()} 토큰")
print(pick.groupby("canonical_dataset").size().to_dict(), "\n")

lay, tra, tim = run(pick.text.tolist(), pick.sample_id.tolist())
RES.mkdir(parents=True, exist_ok=True)
lay.to_csv(RES / "decompx_reconstruction_audit.csv", index=False)
tra.to_csv(RES / "d_conservation_audit.csv", index=False)
tra.to_csv(RES / "projection_identity_audit.csv", index=False)

print("=== 층별 복원 (상대 L2) ===")
print(lay.groupby("layer").rel_l2.agg(["mean", "max"]).to_string(), "\n")
print("=== transition 별 보존/사영/mass 상대오차 (최대) ===")
print(tra.groupby("layer_transition")[["d_rel_err", "proj_rel_err", "mass_rel_err"]]
      .max().to_string(), "\n")
ok = verdict(lay, tra)

tim["bin"] = pd.cut(tim.n_tokens, [0, 32, 64, 128, 256, 512])
print("\n=== DecompX 비용 (길이별) ===")
print(tim.groupby("bin", observed=True).sec_per_sample.agg(["mean", "count"]).round(3).to_string())
tim.to_csv(RES / "decompx_cost.csv", index=False)
print(f"\n판정: {'PHASE A 수치 감사 통과' if ok else '★ 허용오차 초과 -> 추출 중단'}")

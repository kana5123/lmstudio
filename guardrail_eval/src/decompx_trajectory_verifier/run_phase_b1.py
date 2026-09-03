"""PHASE B1 실행: 32 표본으로 C / Y / a / 패딩 계약 감사."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_trajectory_verifier.audit import audit_batch, verdict
from src.decompx_trajectory_verifier.base_adapter import PromptGuard2Adapter
from src.decompx_trajectory_verifier.config import (BASE_MODEL, RES, TOL_ENCODER_REL,
                                                    TOL_HEAD_SCALED, TOL_MARGIN_SCALED)
from src.decompx_trajectory_verifier.decompx_adapter import DecompXAdapter

PRED = Path(__file__).resolve().parents[2] / "data/decompx_verifier/pg2_predictions.parquet"


def main(n=32, device="cuda", chunk=4, dtype=torch.float32):
    base = PromptGuard2Adapter(BASE_MODEL, device, dtype=dtype)
    dxa = DecompXAdapter(base)
    print(f"base: hidden={base.get_hidden_size()} layers={base.get_num_layers()} "
          f"labels={base.get_num_labels()} attack_id={base.get_attack_label_id()} "
          f"benign_id={base.get_benign_label_id()} CLS={base.get_decision_position()} "
          f"max_len={base.get_max_length()}")
    print(f"bias 분해 방식 = {dxa.bias_decomp_mode}  (bias 를 토큰 기여에 분배 -> 재구성에 포함)\n")

    d = pd.read_parquet(PRED, columns=["sample_id", "text", "use", "length_ok",
                                       "token_length", "confusion_cell"])
    d = d[(d.use == "MAIN") & d.length_ok & d.token_length.between(15, 250)]
    pick = pd.concat([d[d.confusion_cell == c].head(n // 4) for c in ("TP", "FP", "TN", "FN")])
    print(f"감사 표본 {len(pick)}개 {pick.confusion_cell.value_counts().to_dict()}")

    E, H, M = [], [], []
    for s in range(0, len(pick), chunk):
        sub = pick.iloc[s:s + chunk]
        enc = base.encode(sub.text.tolist())
        e, h, m, _ = audit_batch(base, dxa, enc["input_ids"], enc["attention_mask"],
                                 sub.sample_id.tolist())
        E += e; H += h; M += m
        torch.cuda.empty_cache()
    E, H, M = pd.DataFrame(E), pd.DataFrame(H), pd.DataFrame(M)
    RES.mkdir(parents=True, exist_ok=True)
    tag = "" if dtype == torch.float32 else "_f64"
    E.to_csv(RES / f"encoder_reconstruction_audit{tag}.csv", index=False)
    H.to_csv(RES / f"head_reconstruction_audit{tag}.csv", index=False)
    M.to_csv(RES / f"margin_reconstruction_audit{tag}.csv", index=False)

    print("\n=== STEP 1  encoder 층별 상대오차 ===")
    print(E.groupby("layer").relative_l2_error.agg(["mean", "max"]).to_string())
    print("\n=== STEP 2  head 클래스별 (sum_k Y_kc vs logit_c) ===")
    print(H.groupby("class_id")[["abs_error", "scaled_error", "rel_error"]]
          .agg(["mean", "max"]).to_string())
    print("   (rel_error 는 개별 로짓으로 나눈 값이라 로짓이 0 을 지나면 발산한다 -- 기록용)")
    print("\n=== STEP 3  signed margin (sum_k a_k vs z_attack-z_benign) ===")
    print(f"  절대오차 평균 {M.abs_error.mean():.3e} 최대 {M.abs_error.max():.3e}")
    print(f"  ||logits|| 정규화 평균 {M.scaled_error.mean():.3e} 최대 {M.scaled_error.max():.3e}")
    print(f"  원 상대오차(기록용) 최대 {M.rel_error.max():.3e}")
    print(f"  margin 범위 {M.margin.min():.3f} ~ {M.margin.max():.3f}\n")
    ok = verdict(E, H, M)

    fails = pd.concat([
        E[E.relative_l2_error > TOL_ENCODER_REL].assign(check="encoder"),
        H[H.scaled_error > TOL_HEAD_SCALED].assign(check="head"),
        M[M.scaled_error > TOL_MARGIN_SCALED].assign(check="margin")], ignore_index=True)
    fails.to_csv(RES / f"reconstruction_failures{tag}.csv", index=False)
    print(f"\n격리된 초과 행 {len(fails)}개 -> {RES/'reconstruction_failures.csv'}")
    print(f"판정: {'PHASE B1 통과' if ok else '★ PHASE B1 실패 -> 추출 중단'}")
    return ok


if __name__ == "__main__":
    import sys as _s
    main(dtype=torch.float64 if "--f64" in _s.argv else torch.float32)

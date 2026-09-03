"""PHASE B1 계약 감사 (§7, §9, §10).

  STEP 1  sum_k C_lk    ~= h_CLS^l          (encoder reconstruction)
  STEP 2  sum_k Y_kc    ~= logit_c          (classification head)
  STEP 3  sum_k a_k     ~= z_attack-z_benign(signed margin)
  STEP 4  패딩 토큰 기여 == 0

허용 오차를 넘는 표본은 통과시키지 않고 격리한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import (EPS, RES, TOL_ENCODER_REL,
                                                    TOL_HEAD_SCALED, TOL_MARGIN_SCALED,
                                                    TOL_PAD_ABS)


def audit_batch(base, dxa, input_ids, attention_mask, sample_ids):
    ev = dxa.extract(input_ids, attention_mask)
    C, Y, a, logits, hs = ev["C"], ev["Y"], ev["a"], ev["logits"], ev["hidden"]
    m = attention_mask.to(C.dtype)
    B, L, T, d = C.shape

    # --- STEP 1: encoder ---------------------------------------------------
    rec = (C * m[:, None, :, None]).sum(2)                 # [B,L,d]
    tgt = torch.stack(hs[1:], 1)[:, :, base.get_decision_position()]   # [B,L,d]
    enc_abs = (rec - tgt).norm(dim=-1)
    enc_rel = enc_abs / (tgt.norm(dim=-1) + EPS)
    enc_max = (rec - tgt).abs().amax(-1)

    # --- STEP 4: 패딩 기여 --------------------------------------------------
    pad = (1 - m)[:, None, :, None]
    padC = (C * pad).abs().amax(dim=(1, 2, 3))             # [B]
    padY = (Y * (1 - m)[:, :, None]).abs().amax(dim=(1, 2))

    # --- STEP 2: head -------------------------------------------------------
    Yrec = (Y * m[:, :, None]).sum(1)                      # [B,C]
    head_abs = (Yrec - logits).abs()
    head_rel = head_abs / (logits.abs() + EPS)             # 기록용(분모가 0 을 지날 수 있음)
    scale = logits.norm(dim=-1, keepdim=True)              # 사라지지 않는 척도
    head_scaled = head_abs / (scale + EPS)

    # --- STEP 3: signed margin ---------------------------------------------
    mg = logits[:, base.get_attack_label_id()] - logits[:, base.get_benign_label_id()]
    arec = (a * m).sum(1)
    mar_abs = (arec - mg).abs()
    mar_rel = mar_abs / (mg.abs() + EPS)                   # 기록용
    mar_scaled = mar_abs / (scale.squeeze(-1) + EPS)

    enc_rows, head_rows, mar_rows = [], [], []
    for b in range(B):
        for l in range(L):
            enc_rows.append(dict(sample_id=sample_ids[b], layer=l + 1,
                                 absolute_l2_error=float(enc_abs[b, l]),
                                 relative_l2_error=float(enc_rel[b, l]),
                                 max_coordinate_error=float(enc_max[b, l]),
                                 pad_contrib_max=float(padC[b])))
        for c in range(logits.shape[1]):
            head_rows.append(dict(sample_id=sample_ids[b], class_id=c,
                                  logit=float(logits[b, c]), sum_Y=float(Yrec[b, c]),
                                  abs_error=float(head_abs[b, c]),
                                  rel_error=float(head_rel[b, c]),
                                  scaled_error=float(head_scaled[b, c]),
                                  logit_norm=float(scale[b, 0]),
                                  pad_contrib_max=float(padY[b])))
        mar_rows.append(dict(sample_id=sample_ids[b], margin=float(mg[b]),
                             sum_a=float(arec[b]), abs_error=float(mar_abs[b]),
                             rel_error=float(mar_rel[b]),
                             scaled_error=float(mar_scaled[b])))
    return enc_rows, head_rows, mar_rows, ev


def verdict(enc, head, mar):
    checks = [("§7  encoder  sum_k C == h_CLS", enc.relative_l2_error.max(), TOL_ENCODER_REL),
              ("§7  패딩 C 기여 == 0", enc.pad_contrib_max.max(), TOL_PAD_ABS),
              ("§9  head     sum_k Y == logit  (/||logits||)", head.scaled_error.max(),
               TOL_HEAD_SCALED),
              ("§9  패딩 Y 기여 == 0", head.pad_contrib_max.max(), TOL_PAD_ABS),
              ("§10 margin   sum_k a == z_a-z_b (/||logits||)", mar.scaled_error.max(),
               TOL_MARGIN_SCALED)]
    ok = True
    print(f"{'검사':<36}{'실측 최대':>12}  {'허용':>9}  판정")
    for nm, got, tol in checks:
        p = got <= tol
        ok &= p
        print(f"{nm:<36}{got:>12.3e}  {tol:>9.0e}  {'통과' if p else '★초과'}")
    return ok

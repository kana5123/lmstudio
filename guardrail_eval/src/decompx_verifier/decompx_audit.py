"""PHASE A 수치 감사: DecompX 복원 / D 보존 / 사영 항등식 / mass 항등식.

기존 검증 결과를 믿지 않고 지금 로드된 PromptGuard2 checkpoint 에서 다시 확인한다.
전부 float32 로 수행한다.  허용 오차를 넘으면 evidence 추출을 시작하지 않는다.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier.config import (EPS, RES, TOL_D_CONSERVE_REL, TOL_MASS_IDENTITY_REL,
                                         TOL_PROJ_IDENTITY_REL, TOL_RECON_REL_L2, load_runtime,
                                         probe_positive_label)
from src.decompx_verifier.sdr import signed_projection, signed_retrieval, transitions
from src.pg2_decompx.decompx_utils import DecompXConfig
from src.pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

DCFG = DecompXConfig(output_all_layers=True, output_encoder=None,
                     include_classifier_w_pooler=False, output_classifier=False)


@torch.no_grad()
def audit_batch(dx, ids, mask, rc):
    """한 배치 감사.  -> 층별 지표 사전."""
    logits, hidden, hs, out = dx.forward(ids, mask, DCFG, output_hidden_states=True)
    C = out.cls_encoder                                  # (B, L, N, d) = C[1]..C[L]
    H = torch.stack(hs, 1)                               # (B, L+1, N, d)
    m = mask.to(C.dtype)

    # --- §8 복원: sum_k C[l,k] == h_CLS[l] --------------------------------
    rec = C.sum(2)                                       # (B, L, d)
    tgt = H[:, 1:, 0]                                    # (B, L, d)
    abs_l2 = (rec - tgt).norm(dim=-1)
    rel_l2 = abs_l2 / (tgt.norm(dim=-1) + EPS)
    max_coord = (rec - tgt).abs().amax(-1)

    # --- 패딩 원천 기여가 정확히 0 인가 -----------------------------------
    pad_contrib = (C * (1 - m)[:, None, :, None]).abs().amax(dim=(1, 2, 3))

    # --- §12 D 보존: sum_k D[l,k] == g_l ----------------------------------
    g, D, H_pre, E = transitions(C, hs, mask)
    gn = g.norm(dim=-1)                                  # (B, K)
    d_abs = (D.sum(2) - g).norm(dim=-1)
    d_rel = d_abs / (gn + EPS)

    # --- §15 사영 항등식: sum_k p == ||g|| --------------------------------
    p = signed_projection(g, D, mask)
    proj_abs = (p.sum(-1) - gn).abs()
    proj_rel = proj_abs / (gn + EPS)

    # --- §18 mass 항등식: mass_pos - mass_neg == ||g|| --------------------
    _, _, mp, mn = signed_retrieval(p, mask)
    mass_rel = (mp - mn - gn).abs() / (gn + EPS)

    return dict(rel_l2=rel_l2, abs_l2=abs_l2, max_coord=max_coord, pad=pad_contrib,
                d_rel=d_rel, d_abs=d_abs, gn=gn, proj_rel=proj_rel, proj_abs=proj_abs,
                mass_rel=mass_rel, mass_pos=mp, mass_neg=mn)


def run(texts, sample_ids, device="cuda", max_tokens_per_batch=4096):
    model, tok, rc = load_runtime(device)
    pos, _, _ = probe_positive_label(model, tok, device)
    rc.positive_label_id = pos
    dx = DecompXDebertaV2(model)

    rows_layer, rows_trans, timing = [], [], []
    order = np.argsort([-len(t) for t in texts])
    i = 0
    while i < len(order):
        # 길이에 맞춰 배치 크기를 정한다 (attribution 이 N^2 로 커진다)
        n_tok = len(tok(texts[order[i]], truncation=False)["input_ids"])
        bs = max(1, min(8, max_tokens_per_batch // max(n_tok, 1)))
        b = order[i:i + bs]; i += bs
        enc = tok([texts[j] for j in b], return_tensors="pt", padding=True,
                  truncation=False).to(device)
        t0 = time.time()
        a = audit_batch(dx, enc["input_ids"], enc["attention_mask"], rc)
        dt = (time.time() - t0) / len(b)
        N = enc["input_ids"].shape[1]
        for bi, j in enumerate(b):
            timing.append(dict(sample_id=sample_ids[j], n_tokens=N, sec_per_sample=dt))
            for l in range(a["rel_l2"].shape[1]):
                rows_layer.append(dict(sample_id=sample_ids[j], layer=l + 1,
                                       abs_l2=float(a["abs_l2"][bi, l]),
                                       rel_l2=float(a["rel_l2"][bi, l]),
                                       max_abs_coord=float(a["max_coord"][bi, l]),
                                       pad_contrib_max=float(a["pad"][bi])))
            for k in range(a["d_rel"].shape[1]):
                rows_trans.append(dict(sample_id=sample_ids[j],
                                       layer_transition=f"L{k+1}->L{k+2}",
                                       g_norm=float(a["gn"][bi, k]),
                                       d_abs_err=float(a["d_abs"][bi, k]),
                                       d_rel_err=float(a["d_rel"][bi, k]),
                                       proj_abs_err=float(a["proj_abs"][bi, k]),
                                       proj_rel_err=float(a["proj_rel"][bi, k]),
                                       mass_rel_err=float(a["mass_rel"][bi, k]),
                                       mass_pos=float(a["mass_pos"][bi, k]),
                                       mass_neg=float(a["mass_neg"][bi, k])))
        del a; torch.cuda.empty_cache()
    return pd.DataFrame(rows_layer), pd.DataFrame(rows_trans), pd.DataFrame(timing)


def verdict(lay, tra):
    """허용 오차 판정.  하나라도 넘으면 False."""
    checks = [
        ("§8  복원 sum_k C == h_CLS", lay.rel_l2.max(), TOL_RECON_REL_L2),
        ("§8  패딩 기여 == 0", lay.pad_contrib_max.max(), 1e-6),
        ("§12 보존 sum_k D == g", tra.d_rel_err.max(), TOL_D_CONSERVE_REL),
        ("§15 사영 sum_k p == ||g||", tra.proj_rel_err.max(), TOL_PROJ_IDENTITY_REL),
        ("§18 mass_pos-mass_neg == ||g||", tra.mass_rel_err.max(), TOL_MASS_IDENTITY_REL),
    ]
    ok = True
    print(f"{'검사':<34}{'실측 최대':>12}  {'허용':>9}  판정")
    for name, got, tol in checks:
        p = got <= tol
        ok &= p
        print(f"{name:<34}{got:>12.3e}  {tol:>9.0e}  {'통과' if p else '★초과'}")
    return ok

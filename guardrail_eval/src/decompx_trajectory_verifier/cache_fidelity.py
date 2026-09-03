"""§22 압축 충실도 검증.

fp32 로 저장한 것을 기준(reference)으로 두고 fp16 / bf16 캐스팅과 비교한다.
캐스팅은 순수한 정밀도 변경이므로 단일 변수 비교가 된다.

C 뿐 아니라 Y, a 도 비교하고, 압축된 값으로 계산한 재구성 항등식
    sum_k Y_kc  ~= logit_c
    sum_k a_k   ~= z_attack - z_benign
의 오차도 정규화·절대값 둘 다 기록한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import EPS, RES

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _stats(ref, cmp_):
    """마지막 축을 벡터로 보고 코사인/상대 L2/절대 오차."""
    r = ref.reshape(-1, ref.shape[-1]).double()
    c = cmp_.reshape(-1, cmp_.shape[-1]).double()
    n = r.norm(dim=-1)
    keep = n > 1e-12
    cos = torch.cosine_similarity(r[keep], c[keep], dim=-1)
    rel = (r[keep] - c[keep]).norm(dim=-1) / n[keep]
    return dict(cos_min=float(cos.min()), cos_mean=float(cos.mean()),
                rel_l2_max=float(rel.max()), rel_l2_mean=float(rel.mean()),
                abs_max=float((r - c).abs().max()))


def compare(ref_blob, attack_id, benign_id):
    """fp32 기준 대비 fp16/bf16 비교표."""
    rows = []
    C32, Y32, a32 = ref_blob["C_flat"].float(), ref_blob["Y_flat"].float(), ref_blob["a_flat"].float()
    logits = ref_blob["logits"].float()
    off = ref_blob["offsets"]
    scale = logits.norm(dim=-1)
    margin = logits[:, attack_id] - logits[:, benign_id]

    def recon(Y, a):
        n = len(off) - 1
        hy = torch.zeros(n, Y.shape[-1], dtype=torch.float64)
        ha = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            s, e = int(off[i]), int(off[i + 1])
            hy[i] = Y[s:e].double().sum(0)
            ha[i] = a[s:e].double().sum()
        return hy, ha

    hy32, ha32 = recon(Y32, a32)
    for name, dt in [("fp32", torch.float32)] + list(DTYPES.items()):
        C, Y, a = C32.to(dt).float(), Y32.to(dt).float(), a32.to(dt).float()
        hy, ha = recon(Y, a)
        he_abs = (hy - logits.double()).abs()
        he_norm = he_abs / (scale.double().unsqueeze(-1) + EPS)
        ma_abs = (ha - margin.double()).abs()
        ma_norm = ma_abs / (scale.double() + EPS)
        r = dict(dtype=name, bytes_per_elem=torch.finfo(dt).bits // 8)
        for tag, ref, cmp_ in (("C", C32, C), ("Y", Y32, Y),
                               ("a", a32.unsqueeze(-1), a.unsqueeze(-1))):
            for k, v in _stats(ref, cmp_).items():
                r[f"{tag}_{k}"] = v
        r |= dict(head_abs_max=float(he_abs.max()), head_norm_max=float(he_norm.max()),
                  head_abs_mean=float(he_abs.mean()), head_norm_mean=float(he_norm.mean()),
                  margin_abs_max=float(ma_abs.max()), margin_norm_max=float(ma_norm.max()),
                  margin_abs_mean=float(ma_abs.mean()), margin_norm_mean=float(ma_norm.mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def compare_configs(ref_blob, attack_id, benign_id):
    """실제 저장 설정별로 재구성 항등식과 encoder 항등식 교란을 잰다.

    구현은 C 만 캐스팅하고 Y/a 는 fp32 로 저장한다.  그 설정이 항등식을 지키는지,
    그리고 Y/a 까지 압축하면 얼마나 깨지는지 나란히 본다.
    """
    C32, Y32, a32 = ref_blob["C_flat"].float(), ref_blob["Y_flat"].float(), ref_blob["a_flat"].float()
    logits, off = ref_blob["logits"].float(), ref_blob["offsets"]
    scale = logits.norm(dim=-1).double()
    margin = (logits[:, attack_id] - logits[:, benign_id]).double()
    n = len(off) - 1

    def sums(C, Y, a):
        hy = torch.zeros(n, Y.shape[-1], dtype=torch.float64)
        ha = torch.zeros(n, dtype=torch.float64)
        hc = torch.zeros(n, C.shape[1], C.shape[2], dtype=torch.float64)
        for i in range(n):
            s, e = int(off[i]), int(off[i + 1])
            hy[i] = Y[s:e].double().sum(0)
            ha[i] = a[s:e].double().sum()
            hc[i] = C[s:e].double().sum(0)          # [L,d] = sum_k C_lk
        return hy, ha, hc

    _, _, hc32 = sums(C32, Y32, a32)
    cn32 = hc32.norm(dim=-1)                        # ||sum_k C_lk|| = ||h_CLS^l||
    rows = []
    cfgs = [("all fp32", torch.float32, torch.float32),
            ("C fp16 + Y/a fp32  (구현)", torch.float16, torch.float32),
            ("C bf16 + Y/a fp32", torch.bfloat16, torch.float32),
            ("all fp16", torch.float16, torch.float16),
            ("all bf16", torch.bfloat16, torch.bfloat16)]
    for nm, cd, yd in cfgs:
        C, Y, a = C32.to(cd).float(), Y32.to(yd).float(), a32.to(yd).float()
        hy, ha, hc = sums(C, Y, a)
        he = (hy - logits.double()).abs()
        me = (ha - margin).abs()
        ce = (hc - hc32).norm(dim=-1)               # encoder 합의 교란량
        rows.append(dict(
            config=nm,
            head_abs_max=float(he.max()), head_norm_max=float((he / (scale[:, None] + EPS)).max()),
            head_abs_mean=float(he.mean()), head_norm_mean=float((he / (scale[:, None] + EPS)).mean()),
            margin_abs_max=float(me.max()), margin_norm_max=float((me / (scale + EPS)).max()),
            margin_abs_mean=float(me.mean()), margin_norm_mean=float((me / (scale + EPS)).mean()),
            encoder_abs_max=float(ce.max()), encoder_norm_max=float((ce / (cn32 + EPS)).max()),
            encoder_abs_mean=float(ce.mean()), encoder_norm_mean=float((ce / (cn32 + EPS)).mean()),
        ))
    return pd.DataFrame(rows)

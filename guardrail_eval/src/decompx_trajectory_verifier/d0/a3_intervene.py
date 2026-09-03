"""§5, §7, §9: 학습된 A3 checkpoint 를 수정하지 않고 추론 시점 개입만 한다.

  branch zeroing : tau_k = 0  또는  e_attr,k = 0
  layer occlusion: 특정 층의 C 를 0 으로 (나머지 층과 depth position 은 그대로)
  hook          : tau_k, v_k, VCLS 를 꺼낸다

개입 결과를 재학습한 A0/A2 와 같은 모델이라고 주장하지 않는다.
기존 A3 예측이 어느 branch/depth 에 의존하는지 진단할 뿐이다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.model import DXTV


def load_a3(ckpt, d=768, L=12, dev="cuda"):
    m = DXTV(d, L, 512, variant="A3", d_v=C.D_V, depth_tf=C.DEPTH_TF, token_tf=C.TOKEN_TF,
             attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT, head_hidden=C.HEAD_HIDDEN).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    return m


@torch.no_grad()
def forward_intervened(m, Cx, f_attr, mask, zero_tau=False, zero_attr=False,
                       occlude_layers=None, want_reps=False):
    """DXTV.forward 를 그대로 재현하되 개입 지점을 연다.  -> dict(logit, tau, v, vcls)"""
    Cc = Cx
    if occlude_layers:
        Cc = Cx.clone()
        for l in occlude_layers:
            Cc[:, l] = 0.0                      # depth embedding 은 proj 뒤에 더해지므로 유지된다
    tau = m.depth(m.proj(m.prepare_C(Cc)))      # [B,T,128]
    if zero_tau:
        tau = torch.zeros_like(tau)
    e = m.anchor(f_attr)                        # [B,T,128]
    if zero_attr:
        e = torch.zeros_like(e)
    v = m.fusion(torch.cat([tau, e], -1))       # [B,T,128]
    z = m.token(v, mask)                        # [B,128]
    logit = m.head(z).squeeze(-1)
    out = dict(logit=logit)
    if want_reps:
        mk = mask.unsqueeze(-1)
        n = mk.sum(1).clamp_min(1)
        out |= dict(tau_mean=(tau * mk).sum(1) / n, v_mean=(v * mk).sum(1) / n, vcls=z)
    return out


def fusion_split(m):
    """§8: Fusion 첫 Linear 의 열을 tau / attr 로 나눈다."""
    W = m.fusion[0].weight.detach()             # [128, 256]
    dv = W.shape[1] // 2
    Wt, Wa = W[:, :dv], W[:, dv:]
    return dict(frob_tau=float(Wt.norm()), frob_attr=float(Wa.norm()),
                ratio_tau_over_attr=float(Wt.norm() / (Wa.norm() + 1e-12)),
                col_l2_tau_mean=float(Wt.norm(dim=0).mean()),
                col_l2_attr_mean=float(Wa.norm(dim=0).mean()))

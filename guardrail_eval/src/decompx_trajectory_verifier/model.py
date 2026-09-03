"""DXTV: DecompX Trajectory Verifier (§11-§19) 와 ablation 변형 (§31).

MAIN evidence 는 정확히 두 가지다.
  C_lk : layer 별 CLS 의 source-token vector decomposition
  Y_kc : classification head 까지 전파한 class 별 token logit 기여 (a_k = Y_attack - Y_benign)

variant:
  A0  final attribution only   -- Y/a 만.  C trajectory 없음
  A1  final layer only         -- C_L 한 층 + Y/a
  A2  trajectory only          -- C 전 층.  Y/a 없음
  A3  MAIN                     -- C 전 층 + Y/a
  A4  delta-C                  -- C 대신 dC_l = C_l - C_(l-1) (l>=2) + Y/a
  A5  token permutation        -- MAIN 과 동일하되 Y/a 의 토큰 인덱스를 섞는다(학습/평가 시)
"""
import torch
import torch.nn as nn

from src.decompx_trajectory_verifier.attribution_anchor import AttributionAnchor
from src.decompx_trajectory_verifier.cell_projector import CellProjector
from src.decompx_trajectory_verifier.depth_trajectory_encoder import DepthTrajectoryEncoder
from src.decompx_trajectory_verifier.token_context_encoder import TokenContextEncoder

VARIANTS = {
    #        use_C  layers      use_attr  delta  perm
    "A0": dict(use_C=False, layer_mode="all",   use_attr=True,  delta=False, perm=False),
    "A1": dict(use_C=True,  layer_mode="last",  use_attr=True,  delta=False, perm=False),
    "A2": dict(use_C=True,  layer_mode="all",   use_attr=False, delta=False, perm=False),
    "A3": dict(use_C=True,  layer_mode="all",   use_attr=True,  delta=False, perm=False),
    "A4": dict(use_C=True,  layer_mode="all",   use_attr=True,  delta=True,  perm=False),
    "A5": dict(use_C=True,  layer_mode="all",   use_attr=True,  delta=False, perm=True),
    # --- PHASE D1A: 층 범위를 처음부터 제한한 variant (§4) --------------------
    # layer_range 는 0-기반 [lo, hi) 이며 L(lo+1) .. L(hi) 를 뜻한다.
    # 바뀌는 것은 depth positional embedding 크기와 depth sequence 길이뿐이다.
    "V0": dict(use_C=False, layer_mode="all", use_attr=True, delta=False, perm=False),
    "V1": dict(use_C=True, layer_mode="range", layer_range=(0, 12), use_attr=True,
               delta=False, perm=False),
    "V2": dict(use_C=True, layer_mode="range", layer_range=(0, 8), use_attr=True,
               delta=False, perm=False),
    "V3": dict(use_C=True, layer_mode="range", layer_range=(8, 12), use_attr=True,
               delta=False, perm=False),
    "V4": dict(use_C=True, layer_mode="range", layer_range=(11, 12), use_attr=True,
               delta=False, perm=False),
}


class DXTV(nn.Module):
    def __init__(self, d, n_layers, max_len, variant="A3", d_v=128, depth_tf=None,
                 token_tf=None, attr_hidden=32, fusion_out=128, head_hidden=64, dropout=0.1):
        super().__init__()
        self.cfgv = dict(VARIANTS[variant]); self.variant = variant
        depth_tf = depth_tf or {}; token_tf = token_tf or {}
        dm = token_tf.get("d_model", 128)
        lm = self.cfgv["layer_mode"]
        if lm == "range":
            lo, hi = self.cfgv["layer_range"]
            assert 0 <= lo < hi <= n_layers, f"잘못된 layer_range {(lo, hi)}"
            L_eff = hi - lo
        else:
            L_eff = {"all": n_layers, "last": 1}[lm]
        if self.cfgv["delta"]:
            L_eff = n_layers - 1 if lm == "all" else 1
        self.L_eff = L_eff

        parts = 0
        if self.cfgv["use_C"]:
            self.proj = CellProjector(d, d_v, L_eff)
            self.depth = DepthTrajectoryEncoder(d_v, **depth_tf)
            parts += depth_tf.get("d_model", 128)
        if self.cfgv["use_attr"]:
            self.anchor = AttributionAnchor(attr_hidden, dm)
            parts += dm
        assert parts > 0, "C 와 Y 를 둘 다 끄면 입력이 없다"
        self.fusion = nn.Sequential(nn.Linear(parts, fusion_out), nn.GELU(), nn.Dropout(dropout))
        tt = {k: v for k, v in token_tf.items() if k != "d_model"}
        assert token_tf.get("d_model", fusion_out) == fusion_out, \
            "token encoder d_model 과 fusion 출력 차원을 맞춘다"
        self.token = TokenContextEncoder(d_model=fusion_out, max_len=max_len, **tt)
        self.head = nn.Sequential(nn.Linear(fusion_out, head_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def prepare_C(self, C):
        """C [B,L,T,d] -> variant 에 맞게 층 선택/차분."""
        if self.cfgv["delta"]:
            C = C[:, 1:] - C[:, :-1]
        lm = self.cfgv["layer_mode"]
        if lm == "last":
            C = C[:, -1:]
        elif lm == "range":
            lo, hi = self.cfgv["layer_range"]
            C = C[:, lo:hi]
        return C

    def forward(self, C, f_attr, mask, perm=None):
        """C [B,L,T,d], f_attr [B,T,3], mask [B,T] -> error_logit [B]"""
        feats = []
        if self.cfgv["use_C"]:
            feats.append(self.depth(self.proj(self.prepare_C(C))))
        if self.cfgv["use_attr"]:
            fa = f_attr
            if self.cfgv["perm"]:
                # §31 A5: C trajectory 와 final attribution 의 토큰 대응을 깨뜨린다.
                # 각각의 주변 분포는 보존된다(같은 값들을 자리만 바꾼다).
                assert perm is not None, "A5 에는 토큰 순열이 필요하다"
                fa = torch.gather(fa, 1, perm.unsqueeze(-1).expand(-1, -1, fa.shape[-1]))
            feats.append(self.anchor(fa))
        v = self.fusion(torch.cat(feats, -1)) if len(feats) > 1 else self.fusion(feats[0])
        return self.head(self.token(v, mask)).squeeze(-1)

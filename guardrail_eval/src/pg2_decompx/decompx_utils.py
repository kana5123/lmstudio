"""DecompX 설정/출력 구조.  원본 third_party/DecompX/src/decompx_utils.py 를 그대로 옮기되,
DeBERTaV2 포팅에서 실제로 지원하는 필드만 남겼다.

원본과 달라진 점(의도적):
  - tanh_approx_type -> act_approx_type.  Prompt Guard 2 의 pooler 활성함수는
    tanh 가 아니라 GELU 다(config.pooler_hidden_act='gelu'). 활성함수 종류에
    무관한 이름으로 바꾸고 zero-origin/linear 근사를 모두 지원한다.
  - aggregation: 원본은 None|'vector'|'rollout'. 우리는 층간 전파가 필수라
    'vector' 만 지원한다(원본 modeling_bert.py:973 — aggregated_encoder_vectors 가
    다음 층의 attribution_vectors 로 들어가는 경로).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class DecompXConfig:
    # --- 편향(bias) 처리 ---
    include_biases: bool = True
    bias_decomp_type: str = "absdot"     # absdot | abssim | norm | equal | cls | dot
    # --- 각 구성요소 포함 여부 ---
    include_LN1: bool = True
    include_FFN: bool = True
    FFN_approx_type: str = "GeLU_ZO"     # GeLU_ZO | GeLU_LA | ReLU
    include_LN2: bool = True
    include_classifier_w_pooler: bool = True
    act_approx_type: str = "ZO"          # ZO | LA  (pooler 활성함수 근사)
    # --- 출력 ---
    aggregation: str = "vector"
    output_all_layers: bool = False
    output_encoder: Optional[str] = None     # None | vector | norm | both
    output_pooler: Optional[str] = None
    output_classifier: bool = True
    # --- 메모리 ---
    ffn_chunk: int = 64      # FFN 분해를 대상토큰 축으로 몇 개씩 끊어 계산할지


@dataclass
class DecompXOutput:
    encoder: Optional[Tuple[torch.Tensor, ...]] = None      # 층별 (B, N, N, H)
    pooler: Optional[torch.Tensor] = None                   # (B, N, H)
    classifier: Optional[torch.Tensor] = None               # (B, N, C)
    cls_encoder: Optional[torch.Tensor] = None              # (B, L, N, H)  CLS 행만


def output_builder(x, mode):
    if x is None or mode is None:
        return None
    if mode == "vector":
        return (x,)
    if mode == "norm":
        return (torch.norm(x, dim=-1),)
    if mode == "both":
        return ((torch.norm(x, dim=-1), x),)
    raise ValueError(mode)

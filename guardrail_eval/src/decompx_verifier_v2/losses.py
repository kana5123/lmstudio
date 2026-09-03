"""§23 분기 손실.  표본은 base 예측에 따라 자기 분기 헤드 하나만 쓴다.

  base_pred == attack : TP -> 0, FP -> 1   (Head_attack)
  base_pred == benign : TN -> 0, FN -> 1   (Head_benign)

base model 손실은 계산하지 않는다.
"""
import torch
import torch.nn.functional as F

CELL_TARGET = {"TP": 0.0, "FP": 1.0, "TN": 0.0, "FN": 1.0}


def branch_loss(out, base_pred, target, pos_weight_attack=None, pos_weight_benign=None):
    """out: DepthVerifier 출력, base_pred [B] (1=attack), target [B] (오답이면 1)."""
    is_atk = base_pred == 1
    loss, n = out["attack_error_logit"].new_zeros(()), 0
    if is_atk.any():
        loss = loss + F.binary_cross_entropy_with_logits(
            out["attack_error_logit"][is_atk], target[is_atk],
            pos_weight=pos_weight_attack, reduction="sum")
        n += int(is_atk.sum())
    if (~is_atk).any():
        loss = loss + F.binary_cross_entropy_with_logits(
            out["benign_error_logit"][~is_atk], target[~is_atk],
            pos_weight=pos_weight_benign, reduction="sum")
        n += int((~is_atk).sum())
    return loss / max(n, 1)


def branch_scores(out, base_pred):
    """평가용: 각 표본의 '오답 확률' 을 해당 분기 헤드에서 가져온다."""
    return torch.where(base_pred == 1, out["attack_error_logit"], out["benign_error_logit"])

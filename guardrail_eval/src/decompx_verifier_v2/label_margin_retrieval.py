"""LabelMarginRetriever: a_lk = q_l^T D_lk 와 부호별 검색(§11-§16).

learned Q/K projection 을 쓰지 않는다.  MLP 를 먼저 거치지 않는다.
sqrt(d) 스케일링도, 코사인 정규화도, q/D 정규화도 하지 않는다.
Taylor 1차 관계 dm ~= q_l^T dh 를 그대로 보존하기 위해서다.

a_lk > 0 : attack margin 증가 방향
a_lk < 0 : benign margin 증가 방향
이를 exact causal contribution 이라고 부르지 않는다.
"""
import torch

EPS = 1e-12


def label_margin_score(q, D, mask):
    """q [B,K,d], D [B,K,T,d], mask [B,T] -> a [B,K,T].  스케일링 없이 순수 내적."""
    a = torch.einsum("bkd,bktd->bkt", q, D)
    return a * mask[:, None, :].to(a.dtype)


def signed_retrieval(a, mask):
    """§14: softmax 가 아니라 ReLU 부호 분리."""
    m = mask[:, None, :].to(a.dtype)
    w_pos, w_neg = torch.relu(a) * m, torch.relu(-a) * m
    M_pos, M_neg = w_pos.sum(-1), w_neg.sum(-1)                     # [B,K]
    al_pos = w_pos / (M_pos.unsqueeze(-1) + EPS)
    al_neg = w_neg / (M_neg.unsqueeze(-1) + EPS)
    # mass 가 0 이면 pooled 출력을 정확히 영벡터로 둔다
    al_pos = al_pos * (M_pos > 0).unsqueeze(-1).to(al_pos.dtype)
    al_neg = al_neg * (M_neg > 0).unsqueeze(-1).to(al_neg.dtype)
    return al_pos, al_neg, M_pos, M_neg


class LabelMarginRetriever:
    """q, D, H_pre -> a, zD_pos/neg, zH_pos/neg, mass_pos/neg."""

    def __call__(self, q, D, H_pre, mask):
        a = label_margin_score(q, D, mask)
        al_pos, al_neg, M_pos, M_neg = signed_retrieval(a, mask)
        pool = lambda al, V: torch.einsum("bkt,bktd->bkd", al, V)
        return dict(a=a,
                    zD_pos=pool(al_pos, D), zD_neg=pool(al_neg, D),
                    zH_pos=pool(al_pos, H_pre), zH_neg=pool(al_neg, H_pre),
                    mass_pos=M_pos, mass_neg=M_neg,
                    alpha_pos=al_pos, alpha_neg=al_neg)

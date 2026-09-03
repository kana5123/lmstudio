"""§31 단위 테스트 T1, T3-T9.  (T2 는 실제 checkpoint 감사에서, T10-T12 는 학습 단계에서)

합성 데이터로 검증할 때는 g 를 C 에서 유도해 만든다(g := sum_k C[l] - sum_k C[l-1]).
그래야 실제 파이프라인과 같은 대수 관계를 재현한다.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_verifier.sdr import (pool, signed_projection, signed_retrieval,
                                      transitions)

B, L, N, d = 3, 12, 17, 32
TOL = 1e-4


def make(seed=0, n_real=None):
    """C 와 은닉상태를 만들되 sum_k C[l,k] == h_CLS[l] 이 성립하게 맞춘다."""
    torch.manual_seed(seed)
    mask = torch.zeros(B, N)
    n_real = n_real if n_real is not None else N
    mask[:, :n_real] = 1.0
    C = torch.randn(B, L, N, d, dtype=torch.float64) * mask[:, None, :, None]
    hs = [torch.randn(B, N, d, dtype=torch.float64)]           # hs[0] 임베딩
    for l in range(L):
        h = torch.randn(B, N, d, dtype=torch.float64)
        h[:, 0] = C[:, l].sum(2 - 1)                            # CLS = sum_k C[l,k]
        hs.append(h)
    return C, hs, mask


def test_T1_shapes():
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    K = L - 1
    assert g.shape == (B, K, d), g.shape
    assert D.shape == (B, K, N, d), D.shape
    assert H.shape == (B, K, N, d), H.shape
    assert E.shape == (B, N, d), E.shape
    p = signed_projection(g, D, mask)
    assert p.shape == (B, K, N), p.shape
    a_pos, a_neg, mp, mn = signed_retrieval(p, mask)
    assert a_pos.shape == (B, K, N) and mp.shape == (B, K)
    print("T1 shapes OK")


def test_T3_sumD_eq_g():
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    err = (D.sum(2) - g).norm(dim=-1) / (g.norm(dim=-1) + 1e-12)
    assert err.max() < TOL, err.max()
    print(f"T3 sum_k D == g   최대 상대오차 {err.max():.3e}")


def test_T4_projection_identity():
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask)
    err = (p.sum(-1) - g.norm(dim=-1)).abs() / (g.norm(dim=-1) + 1e-12)
    assert err.max() < TOL, err.max()
    print(f"T4 sum_k p == ||g||   최대 상대오차 {err.max():.3e}")


def test_T5_mass_identity():
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask)
    _, _, mp, mn = signed_retrieval(p, mask)
    err = (mp - mn - g.norm(dim=-1)).abs() / (g.norm(dim=-1) + 1e-12)
    assert err.max() < TOL, err.max()
    print(f"T5 mass_pos - mass_neg == ||g||   최대 상대오차 {err.max():.3e}")


def test_T6_padding_no_effect():
    """패딩 위치에 쓰레기 값을 넣어도 결과가 같아야 한다."""
    C, hs, mask = make(n_real=11)
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask)
    a_pos, a_neg, mp, mn = signed_retrieval(p, mask)
    z = pool(a_pos, D)

    C2 = C.clone(); C2[:, :, 11:] = torch.randn_like(C2[:, :, 11:]) * 1e3
    hs2 = [h.clone() for h in hs]
    for h in hs2:
        h[:, 11:] = torch.randn_like(h[:, 11:]) * 1e3
    g2, D2, H2, E2 = transitions(C2, hs2, mask)
    p2 = signed_projection(g2, D2, mask)
    a2, _, mp2, _ = signed_retrieval(p2, mask)
    z2 = pool(a2, D2)
    assert (g2 - g).abs().max() < 1e-9, "CLS 이동이 패딩에 오염됐다"
    assert (z2 - z).abs().max() < 1e-9, (z2 - z).abs().max()
    assert (mp2 - mp).abs().max() < 1e-9
    print("T6 패딩은 검색에 영향 없음 OK")


def test_T7_no_counter_component():
    """반대 방향 성분이 없으면 z*_neg 가 정확히 0 이어야 한다."""
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask).abs()          # 전부 양수로 강제
    a_pos, a_neg, mp, mn = signed_retrieval(p, mask)
    assert mn.abs().max() == 0.0, mn.abs().max()
    assert pool(a_neg, D).abs().max() == 0.0
    print("T7 반대 성분 없으면 z_neg == 0 OK")


def test_T8_joint_permutation_invariant():
    """D/H/E/mask 를 같은 순열로 바꾸면 pooled 증거는 변하지 않아야 한다(§21)."""
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask)
    a_pos, a_neg, mp, mn = signed_retrieval(p, mask)
    base = [pool(a_pos, D), pool(a_neg, D), pool(a_pos, H), pool(a_pos, E), mp, mn]

    torch.manual_seed(7)
    perm = torch.randperm(N)
    Dp, Hp, Ep, mkp = D[:, :, perm], H[:, :, perm], E[:, perm], mask[:, perm]
    pp = signed_projection(g, Dp, mkp)
    ap, an, mpp, mnp = signed_retrieval(pp, mkp)
    got = [pool(ap, Dp), pool(an, Dp), pool(ap, Hp), pool(ap, Ep), mpp, mnp]
    for a, b in zip(base, got):
        assert (a - b).abs().max() < 1e-9, (a - b).abs().max()
    print("T8 동시 순열 불변 OK (V1 의 의도된 성질)")


def test_T9_permuting_D_alone_changes_retrieval():
    """D 만 섞고 H/E 를 두면 검색되는 내용이 달라져야 한다."""
    C, hs, mask = make()
    g, D, H, E = transitions(C, hs, mask)
    p = signed_projection(g, D, mask)
    a_pos, _, _, _ = signed_retrieval(p, mask)
    zH = pool(a_pos, H)

    torch.manual_seed(11)
    perm = torch.randperm(N)
    p2 = signed_projection(g, D[:, :, perm], mask)
    a2, _, _, _ = signed_retrieval(p2, mask)
    zH2 = pool(a2, H)          # H 는 그대로, alpha 만 바뀜
    diff = (zH - zH2).norm(dim=-1) / (zH.norm(dim=-1) + 1e-12)
    assert diff.mean() > 0.1, f"D 순열이 검색을 바꾸지 못했다 (평균 상대차 {diff.mean():.4f})"
    print(f"T9 D 단독 순열은 검색을 바꿈  평균 상대차 {diff.mean():.3f}")


if __name__ == "__main__":
    for f in [test_T1_shapes, test_T3_sumD_eq_g, test_T4_projection_identity,
              test_T5_mass_identity, test_T6_padding_no_effect, test_T7_no_counter_component,
              test_T8_joint_permutation_invariant, test_T9_permuting_D_alone_changes_retrieval]:
        f()
    print("\n합성 데이터 단위 테스트 전부 통과")

"""B/C/D/E. 분해 복원 오차 · 패딩 오염 · 배치 동등성.

복원 항등식(편향 포함 규약):
  DecompX 의 bias_decomposer 는 편향을 원천토큰들에게 **정규화 가중치로 나눠** 더한다.
  따라서 편향은 별도 항으로 남지 않고 각 토큰 기여 안에 흡수된다.  결과적으로

      Σ_k C_cls^(l)[k]  ==  h_cls^(l)          (l = 1..L, 편향 별도 항 없음)
      Σ_k A[k, c]       ==  logits[c]

  이 항등식은 **수학적으로 정확**하다.  남는 오차는 float 누적오차뿐이므로,
  float64 로 올리면 오차가 함께 줄어드는지로 '근사'와 '반올림'을 구분한다.
"""
import torch
from _common import load, encode, TEXTS
from pg2_decompx.decompx_utils import DecompXConfig

DCFG = DecompXConfig(output_all_layers=True, output_encoder="vector", output_classifier=True)


def _run(dtype=torch.float32, texts=TEXTS):
    tok, m, wrap = load(dtype)
    enc = encode(tok, texts)
    logits, hidden, hs, out = wrap.forward(enc["input_ids"], enc["attention_mask"], DCFG,
                                           output_hidden_states=True)
    return enc, logits, hidden, hs, out


def test_hidden_reconstruction():
    """B. 각 층에서 토큰 기여의 합 == 실제 CLS 은닉표현."""
    for dtype in (torch.float32, torch.float64):
        enc, logits, hidden, hs, out = _run(dtype)
        L = out.cls_encoder.shape[1]
        worst = 0.0
        print(f"--- dtype={dtype} ---")
        for l in range(L):
            rec = out.cls_encoder[:, l].sum(1)          # (B,H)
            ref = hs[l + 1][:, 0]                       # hidden_states[l+1] = l번째 층 출력
            e = (rec - ref).abs().max().item()
            rel = e / ref.abs().max().item()
            worst = max(worst, rel)
            print(f"  layer {l+1:2}  max_abs={e:.3e}  rel={rel:.3e}")
        assert worst < 1e-3, worst
        if dtype == torch.float64:
            assert worst < 1e-10, f"float64 에서도 큰 오차 -> 구조적 근사 의심: {worst}"


def test_classifier_reconstruction():
    """C. 토큰별 분류기 기여의 합 == 실제 로짓."""
    for dtype in (torch.float32, torch.float64):
        enc, logits, hidden, hs, out = _run(dtype)
        rec = out.classifier.sum(1)
        e = (rec - logits).abs().max().item()
        print(f"C dtype={dtype}  max_abs={e:.3e}")
        assert e < 1e-3
        if dtype == torch.float64:
            assert e < 1e-9, e


def test_padding_contributions_are_zero():
    """D. 패딩 토큰의 기여가 정확히 0 인가.

    DeBERTaV2 임베딩은 마스크를 곱해 패딩 임베딩을 0 으로 만들고
    (modeling_deberta_v2.py:566-574), 주의집중 마스크가 패딩 '열'을 막는다.
    두 장치가 함께 작동하면 패딩 토큰은 CLS 에 기여할 수 없어야 한다.
    """
    enc, logits, hidden, hs, out = _run()
    mask = enc["attention_mask"]                        # (B,N)
    pad = mask == 0
    assert pad.any(), "패딩이 없는 배치로는 이 테스트가 무의미하다"
    contrib = out.cls_encoder[:, -1]                    # (B,N,H)
    pad_norm = contrib[pad].norm(dim=-1)
    real_norm = contrib[~pad].norm(dim=-1)
    print(f"D 패딩 기여 노름 max={pad_norm.max():.3e} / 실토큰 기여 노름 중앙값={real_norm.median():.3e}")
    cls_pad = out.classifier[pad].abs()
    print(f"  분류기 기여(패딩) max={cls_pad.max():.3e}")
    assert pad_norm.max() < 1e-4, pad_norm.max()
    assert cls_pad.max() < 1e-5, cls_pad.max()


def test_batch_equivalence():
    """E. 배치 1 과 배치 N 에서 같은 문장의 결과가 같은가.

    **상대오차로 판정한다.**  절대오차로 보면 안 되는 이유: 토큰 기여 벡터는 서로
    상쇄되면서 최종 표현을 만들기 때문에 개별 기여의 크기가 결과보다 훨씬 클 수 있다
    (실측: 3토큰 입력에서 기여 최대 1882, 최종 은닉은 O(1)).  또한 HuggingFace 원본
    자체도 패딩 길이가 바뀌면 float32 에서 2e-5~8e-5 만큼 흔들린다(실측).  즉 이
    편차는 우리 구현의 결함이 아니라 부동소수점 누적 순서 차이다.
    """
    tok, m, wrap = load()
    encN = encode(tok, TEXTS)
    with torch.no_grad():
        refN = m(**encN, output_hidden_states=True)
    _, _, _, outN = wrap.forward(encN["input_ids"], encN["attention_mask"], DCFG)
    worst_h = worst_c = 0.0
    for i, t in enumerate(TEXTS):
        e1 = encode(tok, [t])
        n = e1["input_ids"].shape[1]
        with torch.no_grad():
            ref1 = m(**e1, output_hidden_states=True)
        hf = (ref1.hidden_states[-1][0, 0] - refN.hidden_states[-1][i, 0]).abs().max().item()
        _, _, _, o1 = wrap.forward(e1["input_ids"], e1["attention_mask"], DCFG)
        c1, cN = o1.cls_encoder[0, -1], outN.cls_encoder[i, -1, :n]
        dh = (c1 - cN).abs().max().item() / c1.abs().max().item()
        a1, aN = o1.classifier[0], outN.classifier[i, :n]
        dc = (a1 - aN).abs().max().item() / a1.abs().max().item()
        worst_h, worst_c = max(worst_h, dh), max(worst_c, dc)
        print(f"E [{i}] len={n:3}  HF원본편차={hf:.2e} | 은닉기여 상대={dh:.3e}  분류기기여 상대={dc:.3e}")
    assert worst_h < 1e-3, worst_h
    assert worst_c < 1e-3, worst_c


if __name__ == "__main__":
    test_hidden_reconstruction(); test_classifier_reconstruction()
    test_padding_contributions_are_zero(); test_batch_equivalence()
    print("PASS")

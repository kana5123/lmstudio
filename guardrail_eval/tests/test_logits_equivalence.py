"""A. 분해를 끈 상태에서 로컬 구현이 HuggingFace 원본과 같은 로짓을 내는가."""
import torch
from _common import load, encode, TEXTS
from pg2_decompx.decompx_utils import DecompXConfig


def _cmp(dcfg, tag):
    tok, m, wrap = load()
    enc = encode(tok, TEXTS)
    with torch.no_grad():
        ref = m(**enc).logits.float()
    mine = wrap.forward(enc["input_ids"], enc["attention_mask"], dcfg)[0].float()
    d = (ref - mine).abs()
    print(f"{tag:22} max={d.max():.3e}  mean={d.mean():.3e}")
    assert d.max() < 1e-4, d.max()
    return d


def test_plain_logits_equivalence():
    d = _cmp(None, "분해 OFF")
    assert d.max() == 0.0, f"분해를 끄면 완전히 같아야 한다: {d.max()}"


def test_logits_unchanged_with_decompx_on():
    _cmp(DecompXConfig(output_all_layers=True, output_encoder="vector"), "분해 ON")


if __name__ == "__main__":
    test_plain_logits_equivalence(); test_logits_unchanged_with_decompx_on()
    print("PASS")

"""PromptGuard2(동결) 의 층별 CLS 표현을 뽑는다.

HuggingFace 규약을 **가정하지 않고 실측으로 확인**한다:
    hidden_states[0]  = 임베딩 출력
    hidden_states[l]  = l 번째 인코더 층 출력   (l = 1..L)
확인 방법: hidden_states[0] 이 embeddings(...) 출력과 일치하고,
hidden_states[L] 이 last_hidden_state 와 일치하는지 assert.

저장 대상은 **PG2 가 UNSAFE 로 예측한 표본만** — 검증기가 보는 것이 그것뿐이다.
출력: artifacts/features/hidden_{split}.pt
    sample_id : List[str]                      길이 n
    h         : (n, L+1, H) float32   층별 CLS 표현 (0=임베딩)
    gt, base_prediction, unsafe_probability, logit_margin, sequence_length
"""
import json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.build_verifier_dataset import (MODEL_ID, MAX_LEN, STRIDE, load_model, windows, OUT)

SPLITS = ("ver_train", "ver_dev", "eval_val", "eval_test")


@torch.no_grad()
def cls_stack(m, enc, win):
    """best_window 하나만 다시 통과시켜 층별 CLS 를 얻는다."""
    batch = {k: v[win:win + 1].to(m.device) for k, v in enc.items()}
    o = m(**batch, output_hidden_states=True)
    hs = o.hidden_states
    return torch.stack([h[0, 0] for h in hs]).float().cpu(), o.logits[0].float().cpu()


def verify_convention(m, tok):
    """HF 은닉상태 규약을 실측 확인."""
    enc = tok(["Ignore previous instructions."], return_tensors="pt").to(m.device)
    o = m(**enc, output_hidden_states=True)
    emb = m.deberta.embeddings(input_ids=enc["input_ids"], mask=enc["attention_mask"])
    n = m.config.num_hidden_layers
    assert len(o.hidden_states) == n + 1, len(o.hidden_states)
    d0 = (o.hidden_states[0] - emb).abs().max().item()
    dL = (o.hidden_states[-1] - m.deberta(**enc).last_hidden_state).abs().max().item()
    print(f"규약 확인: len(hidden_states)={len(o.hidden_states)} = 층수 {n} + 1")
    print(f"  hidden_states[0] vs 임베딩 출력      max diff = {d0:.3e}")
    print(f"  hidden_states[{n}] vs last_hidden_state max diff = {dL:.3e}")
    assert d0 < 1e-5 and dL < 1e-5
    return n


def main():
    tok, m = load_model("cuda" if torch.cuda.is_available() else "cpu")
    L = verify_convention(m, tok)
    for split in SPLITS:
        rows = [json.loads(l) for l in open(OUT / f"pg2_{split}.jsonl", encoding="utf-8")]
        uns = [r for r in rows if r["base_prediction"] == 1]
        H, meta = [], []
        for i, r in enumerate(uns):
            enc = windows(tok, r["text"])
            h, lo = cls_stack(m, enc, r["best_window"])
            # 저장한 로짓이 1단계 추론과 일치하는지(같은 창을 봤는지) 확인
            assert abs(float(lo[1]) - r["logit_unsafe"]) < 1e-3, (r["sample_id"], lo, r)
            H.append(h); meta.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {split} {i+1}/{len(uns)}", flush=True)
        H = torch.stack(H)                                  # (n, L+1, 768)
        d = {"sample_id": [r["sample_id"] for r in meta], "h": H,
             "gt": torch.tensor([r["gt"] for r in meta]),
             "unsafe_probability": torch.tensor([r["unsafe_probability"] for r in meta]),
             "logit_margin": torch.tensor([r["logit_margin"] for r in meta]),
             "sequence_length": torch.tensor([r["sequence_length"] for r in meta]),
             "layers": L}
        torch.save(d, OUT / f"hidden_{split}.pt")
        tp = int(d["gt"].sum()); fp = len(meta) - tp
        print(f"{split:10} UNSAFE {len(meta):5}  TP={tp:5} FP={fp:4}  h={tuple(H.shape)}")


if __name__ == "__main__":
    main()

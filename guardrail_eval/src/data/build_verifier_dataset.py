"""PromptGuard2(동결) 를 전 분할에 추론해 verifier 데이터셋 메타데이터를 만든다.

기존 벤치마크(guards.py:SeqClassifierGuard.score)와 **같은 채점 규칙**을 쓴다:
512 토큰 창을 stride 128 로 겹쳐 훑고 창별 위험도의 최댓값을 문서 점수로 삼는다.
DecompX 는 창 하나에만 돌릴 수 있으므로, 최댓값을 낸 창의 인덱스(best_window)를
같이 저장해 뒤 단계가 정확히 그 창을 재현하도록 한다.

라벨 방향은 실측으로 확인했다: LABEL_1 = MALICIOUS(UNSAFE), LABEL_0 = BENIGN.
(공격문 p1=0.9996, 정상문 p1=0.0004 — 2026-09-01 확인)
"""
import json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.splits import build, assert_no_overlap

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MAX_LEN, STRIDE, WIN_BATCH = 512, 128, 64
UNSAFE_ID, BENIGN_ID = 1, 0          # 실측 확인값
OUT = Path(__file__).resolve().parents[2] / "artifacts" / "features"


def load_model(device="cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)                      # frozen 명시
    assert m.config.id2label == {0: "LABEL_0", 1: "LABEL_1"}, m.config.id2label
    return tok, m


def windows(tok, text):
    """겹치는 512토큰 창. guards.py:334-338 과 동일한 인자."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, padding=True)
    enc.pop("overflow_to_sample_mapping", None)
    return enc


@torch.no_grad()
def score_one(tok, m, text):
    enc = windows(tok, text)
    n_win = enc["input_ids"].shape[0]
    logits = []
    for k in range(0, n_win, WIN_BATCH):
        chunk = {kk: v[k:k + WIN_BATCH].to(m.device) for kk, v in enc.items()}
        logits.append(m(**chunk).logits.float().cpu())
    logits = torch.cat(logits)                        # (n_win, 2)
    probs = logits.softmax(-1)
    best = int(probs[:, UNSAFE_ID].argmax())
    lo = logits[best]
    return dict(
        unsafe_probability=float(probs[best, UNSAFE_ID]),
        benign_probability=float(probs[best, BENIGN_ID]),
        logit_unsafe=float(lo[UNSAFE_ID]), logit_benign=float(lo[BENIGN_ID]),
        logit_margin=float(lo[UNSAFE_ID] - lo[BENIGN_ID]),
        n_windows=n_win, best_window=best,
        sequence_length=int(enc["attention_mask"][best].sum()),
        total_tokens=int(len(tok(text, add_special_tokens=True)["input_ids"])),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, m = load_model(device)
    splits = build()
    assert_no_overlap(splits)
    OUT.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        recs = []
        for i, r in enumerate(rows):
            s = score_one(tok, m, r["text"])
            s["base_prediction"] = int(s["unsafe_probability"] >= 0.5)
            recs.append({**r, **s})
            if (i + 1) % 2000 == 0:
                print(f"  {name} {i+1}/{len(rows)}", flush=True)
        p = OUT / f"pg2_{name}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs),
                     encoding="utf-8")
        u = [x for x in recs if x["base_prediction"] == 1]
        tp = sum(x["gt"] == 1 for x in u); fp = len(u) - tp
        print(f"{name:10} n={len(recs):6} UNSAFE예측={len(u):5} (TP={tp:5} FP={fp:4})  -> {p.name}")


if __name__ == "__main__":
    main()

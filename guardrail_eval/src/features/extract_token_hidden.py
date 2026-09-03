"""토큰별(‘CLS만’이 아니라) 은닉표현 h^(1), h^(L) 을 뽑는다 — 증류 학생의 입력.

DecompX 는 배포에 쓰기엔 너무 느리다(실측 189배).  대신 **PG2 가 어차피 계산하는**
평범한 은닉표현에서 DecompX 신호를 예측하는 작은 머리(head)를 학습할 것이므로,
그 입력이 되는 토큰별 은닉표현을 저장한다.  추가 비용은 output_hidden_states 뿐이다.
"""
import argparse, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.build_verifier_dataset import load_model, windows, OUT

MAXN = 512


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True)
    a = ap.parse_args()
    tok, m = load_model("cuda" if torch.cuda.is_available() else "cpu")
    L = m.config.num_hidden_layers
    rows = [json.loads(l) for l in open(OUT / f"pg2_{a.split}.jsonl", encoding="utf-8")]
    uns = [r for r in rows if r["base_prediction"] == 1]
    N = len(uns)
    H1 = torch.zeros(N, MAXN, 768, dtype=torch.float16)
    HL = torch.zeros(N, MAXN, 768, dtype=torch.float16)
    for i, r in enumerate(uns):
        enc = windows(tok, r["text"]); w = r["best_window"]
        b = {k: v[w:w + 1].to(m.device) for k, v in enc.items()}
        hs = m(**b, output_hidden_states=True).hidden_states
        n = b["attention_mask"].sum().item()
        H1[i, :n] = hs[1][0, :n].half().cpu(); HL[i, :n] = hs[L][0, :n].half().cpu()
        if (i + 1) % 1000 == 0:
            print(f"  {a.split} {i+1}/{N}", flush=True)
    torch.save({"sample_id": [r["sample_id"] for r in uns], "h1": H1, "hL": HL},
               OUT / f"tokenhidden_{a.split}.pt")
    print(f"{a.split}: {N}건 저장  h1={tuple(H1.shape)}")


if __name__ == "__main__":
    main()

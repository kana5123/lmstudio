"""a 가 어디에 몰려 있나 — 특수토큰([CLS]/[SEP]) vs 내용 토큰 (지시문 17절 보강).

sum_k a_k^(l) = dot(v,g^(l)) = p^(l) 이 성립하므로(Q6 확인됨),
"토큰 수준 신호"가 실제로는 몇 개 토큰에 몰려 있는지 재는 것이 의미 있다.
[CLS]/[SEP] 가 합의 대부분을 차지하면 '토큰별 설명'은 사실상 '특수토큰 설명'이다.
"""
import csv, glob, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/directional_alignment"
RES = ROOT / "results/directional_alignment"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"


def main(split="eval_test"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    SPECIAL = set(tok.all_special_ids)
    fs = sorted(glob.glob(str(ART / f"dir_{split}_*of*.pt")))
    ds = [torch.load(f, weights_only=False) for f in fs]
    A = torch.cat([d["a"] for d in ds]).numpy()
    M = torch.cat([d["mask"] for d in ds]).numpy()
    ID = torch.cat([d["input_ids"] for d in ds]).numpy()
    y = torch.cat([d["gt"] for d in ds]).numpy()
    P = torch.cat([d["p"] for d in ds]).numpy()
    T = ds[0]["n_transitions"]
    trans = [f"L{l+1}->L{l+2}" for l in range(T)]
    sp = np.isin(ID, list(SPECIAL)) & M           # 특수토큰이면서 유효
    ct = M & ~sp                                  # 내용(비특수) 토큰

    rows = []
    print(f"{split}: n={len(y)} TP={int(y.sum())} FP={int((1-y).sum())}")
    print("\n=== sum_k a_k 중 [CLS]/[SEP] 가 차지하는 비중, 그리고 그것만으로의 판별력 ===")
    print(f"{'전이':10} {'라벨':4} {'특수토큰 몫':>12} {'내용토큰 몫':>12} | "
          f"{'AUROC(특수만)':>13} {'AUROC(내용만)':>13} {'AUROC(전체 p)':>13}")
    from sklearn.metrics import roc_auc_score
    for l in range(T):
        a = A[:, l]
        s_sum = (a * sp).sum(1)
        c_sum = (a * ct).sum(1)
        tot = P[:, l]
        au_s = roc_auc_score(y, s_sum); au_c = roc_auc_score(y, c_sum)
        au_t = roc_auc_score(y, tot)
        for g, lbl in ((1, "TP"), (0, "FP")):
            m = y == g
            # 몫 = |특수 합| / (|특수 합| + |내용 합|)  — 부호 상쇄를 피해 절대값 기준
            den = np.abs(s_sum[m]) + np.abs(c_sum[m]) + 1e-12
            share_s = float(np.mean(np.abs(s_sum[m]) / den))
            share_c = float(np.mean(np.abs(c_sum[m]) / den))
            rows.append({"transition": trans[l], "label": lbl,
                         "special_share_of_sum": share_s, "content_share_of_sum": share_c,
                         "auroc_special_only": au_s, "auroc_content_only": au_c,
                         "auroc_full_p": au_t,
                         "n_special_tokens_mean": float(sp[m].sum(1).mean()),
                         "n_content_tokens_mean": float(ct[m].sum(1).mean())})
            print(f"{trans[l]:10} {lbl:4} {share_s*100:11.1f}% {share_c*100:11.1f}% | "
                  f"{au_s:13.4f} {au_c:13.4f} {au_t:13.4f}")
    with open(RES / "special_vs_content_concentration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n저장 -> {RES/'special_vs_content_concentration.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_test")

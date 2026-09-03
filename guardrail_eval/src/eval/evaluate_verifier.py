"""검증기 자체 성능 표 (지시문 12·13장의 검증기 부분).

절제 실행 결과를 씨앗 평균±표준편차로 모으고, 저FPR 운용점에서의 분리력도 같이 낸다.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from analysis.metrics import auroc, auprc, pick_threshold, recall_fpr

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts/results"


def main():
    S = torch.load(RES / "verifier_scores.pt", weights_only=False)
    runs = json.loads((RES / "ablation_runs.json").read_text())
    names = sorted({k.rsplit("_s", 1)[0] for k in S})
    rows = []
    for n in names:
        ks = sorted(k for k in S if k.rsplit("_s", 1)[0] == n)
        m = {}
        for split in ("ver_dev", "eval_val", "eval_test"):
            au = [auroc(np.array(S[k][split]["y"]), np.array(S[k][split]["score"])) for k in ks]
            ap = [auprc(np.array(S[k][split]["y"]), np.array(S[k][split]["score"])) for k in ks]
            m[f"{split}_auroc"] = (float(np.mean(au)), float(np.std(au)))
            m[f"{split}_auprc"] = (float(np.mean(ap)), float(np.std(ap)))
        # 검증기 관점의 저FPR: '오탐을 몇 %만 남기면서 정탐을 몇 % 지키나'
        r = []
        for k in ks:
            vs, vy = np.array(S[k]["eval_val"]["score"]), np.array(S[k]["eval_val"]["y"])
            ts, ty = np.array(S[k]["eval_test"]["score"]), np.array(S[k]["eval_test"]["y"])
            thr = pick_threshold(vs.tolist(), vy.tolist(), 0.05)   # FP 5%만 통과
            rec, fpr = recall_fpr(ts.tolist(), ty.tolist(), thr)
            r.append((rec, fpr))
        m["tp_keep@fp5pct"] = (float(np.mean([x[0] for x in r])), float(np.std([x[0] for x in r])))
        m["fp_pass@fp5pct"] = (float(np.mean([x[1] for x in r])), float(np.std([x[1] for x in r])))
        p = [x["params"] for x in runs["runs"] if x["name"] == n]
        m["params"] = p[0] if p else None
        rows.append({"name": n, **m})
        print(f"{n:22} params={str(m['params']):>9}  "
              f"test AUROC={m['eval_test_auroc'][0]:.4f}±{m['eval_test_auroc'][1]:.4f}  "
              f"AUPRC={m['eval_test_auprc'][0]:.4f}  "
              f"TP유지@FP5%={m['tp_keep@fp5pct'][0]:.4f}±{m['tp_keep@fp5pct'][1]:.4f}")
    print(f"\nB0 (PG2 점수만): " + "  ".join(
        f"{r['split']} AUROC={r['auroc']:.4f}" for r in runs["b0"]))
    (RES / "verifier_table.json").write_text(json.dumps(
        {"rows": rows, "b0": runs["b0"]}, ensure_ascii=False, indent=1))
    print(f"저장 -> {RES/'verifier_table.json'}")


if __name__ == "__main__":
    main()

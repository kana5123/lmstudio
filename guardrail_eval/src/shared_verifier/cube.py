"""PHASE1-4: 모델 x 데이터셋 x 혼동셀 큐브.

각 (모델, 데이터셋) 마다 TP/FP/TN/FN 개수와 파생 지표를 낸다.
공유 검증기가 학습 가능한지는 결국 "오답 셀(FP,FN)이 충분히 있는가" 에 달려 있으므로
셀 개수 자체가 1차 관문이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HID = ROOT / "artifacts/shared_verifier/hidden"
RES = ROOT / "results/shared_verifier"


def main():
    rows = []
    for f in sorted(HID.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        cell = np.array(d["cell"])
        c = {k: int((cell == k).sum()) for k in ("TP", "FP", "TN", "FN")}
        n = len(cell)
        pos, neg = c["TP"] + c["FN"], c["TN"] + c["FP"]
        rows.append(dict(
            model=d["model"], dataset=d["dataset"], attack_family=d["attack_family"],
            compatible=d["compatible"], n=n, layers=d["layers"], **c,
            n_correct=c["TP"] + c["TN"], n_wrong=c["FP"] + c["FN"],
            acc=(c["TP"] + c["TN"]) / n,
            tpr=c["TP"] / pos if pos else np.nan,
            fpr=c["FP"] / neg if neg else np.nan,
            wrong_rate=(c["FP"] + c["FN"]) / n,
            p_attack_mean=float(d["p_attack"].mean()),
        ))
    df = pd.DataFrame(rows).sort_values(["model", "dataset"])
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "confusion_cube.csv", index=False)

    print(f"쌍 {len(df)}개, 총 샘플 {df.n.sum():,}\n")
    piv = df.pivot_table(index="dataset", columns="model", values="wrong_rate")
    print("오답률 (FP+FN)/n  -- 낮을수록 그 모델이 그 데이터셋을 잘 맞춤")
    print(piv.round(3).to_string(), "\n")
    print("모델별 합계")
    g = df.groupby("model")[["n", "TP", "FP", "TN", "FN", "n_wrong"]].sum()
    g["wrong_rate"] = g.n_wrong / g.n
    print(g.to_string(), "\n")
    thin = df[df.n_wrong < 50]
    if len(thin):
        print(f"오답 50개 미만이라 검증기 학습에 쓰기 어려운 쌍 {len(thin)}개:")
        print(thin[["model", "dataset", "n", "n_wrong"]].to_string(index=False))
    print(f"\n저장 -> {RES/'confusion_cube.csv'}")


if __name__ == "__main__":
    main()

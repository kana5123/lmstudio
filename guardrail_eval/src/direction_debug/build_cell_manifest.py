"""2x2 혼동 셀 매니페스트 (지시문 4·12·13절).

네 cell(TP/FP/TN/FN)을 모두 가진 데이터셋만 모은다.
cell 당 표본을 CAP 으로 제한한다 — cell **평균**만 필요하므로 3,000이면 충분하고,
그 사실과 상한을 표에 명시한다.

데이터셋 정체:
  A wildjailbreak:adversarial   같은 WildTeaming 절차, 말뭉치 교란 없음
  B promptshield:test           PromptShield 평가 split
  C piguard:Question Set        ★ FP 라벨 오염 확인됨 — 결과 해석 시 반드시 병기
  D promptshield:train          같은 데이터셋의 다른 split (독립 데이터셋 아님, 내부 대조용)
  Z jailbreaksovertime          ★ 말뭉치 교란 확정 — 독립 데이터셋으로 세지 않고 대조군
"""
import hashlib, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "data/multisource_guard"
FEAT = ROOT / "artifacts/features"
OUT = ROOT / "data/direction_debug"
CAP = 3000
SEED = 0

GROUPS = {
    "wildjailbreak": "wildjailbreak:adversarial",
    "promptshield_test": "promptshield:test",
    "questionset": "piguard:Question Set",
    "promptshield_train": "promptshield:train",
}


def cells_from(gt_unsafe, pred_unsafe):
    return np.where(gt_unsafe & pred_unsafe, "TP",
           np.where(~gt_unsafe & pred_unsafe, "FP",
           np.where(~gt_unsafe & ~pred_unsafe, "TN", "FN")))


def jot_rows():
    """JailbreaksOverTime — 보존된 pg2_*.jsonl 에서 네 cell 을 만든다."""
    rows = []
    for split in ("ver_train", "ver_dev", "eval_val", "eval_test"):
        for l in open(FEAT / f"pg2_{split}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows.append({"sample_id": r["sample_id"], "text": r["text"],
                         "gt_unsafe": bool(r["gt"] == 1),
                         "pred_unsafe": bool(r["base_prediction"] == 1),
                         "orig_split": split, "duplicate_group_id": r["sample_id"]})
    d = pd.DataFrame(rows)
    d["dataset"] = "jailbreaksovertime"
    d["confusion_cell"] = cells_from(d["gt_unsafe"].values, d["pred_unsafe"].values)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cc = pd.read_parquet(MS / "confusion_cells.parquet")
    can = pd.read_parquet(MS / "canonical_samples.parquet")[["sample_id", "text"]]
    sm = pd.read_parquet(MS / "split_manifest.parquet")[["sample_id", "split", "group_key"]]
    ms = cc.merge(can, on="sample_id").merge(sm, on="sample_id")

    parts = []
    for name, grp in GROUPS.items():
        g = ms[ms["source_group"] == grp].copy()
        g["dataset"] = name
        g["gt_unsafe"] = g["binary_main_label"] == "UNSAFE"
        g["pred_unsafe"] = g["pg2_prediction"] == "UNSAFE"
        g["orig_split"] = g["split"]
        parts.append(g[["sample_id", "text", "gt_unsafe", "pred_unsafe", "orig_split",
                        "duplicate_group_id", "dataset", "confusion_cell"]])
    j = jot_rows()
    # JOT 는 train/held-out 구분이 필요하다: ver_train = 방향 적합용, 나머지 = held-out
    j["split_role"] = np.where(j["orig_split"] == "ver_train", "train", "heldout")
    for p in parts:
        p["split_role"] = np.where(p["orig_split"] == "train", "train", "heldout")
    allr = pd.concat(parts + [j[["sample_id", "text", "gt_unsafe", "pred_unsafe", "orig_split",
                                 "duplicate_group_id", "dataset", "confusion_cell",
                                 "split_role"]]], ignore_index=True)

    print("=== 상한 적용 전 cell 수 ===")
    t = pd.crosstab([allr["dataset"], allr["split_role"]], allr["confusion_cell"])
    print(t.to_string())

    # cell x split_role 별로 CAP 표본 (그룹 단위가 아니라 행 단위 — 평균 추정용)
    rng = np.random.default_rng(SEED)
    keep = []
    for (ds, sr, ce), g in allr.groupby(["dataset", "split_role", "confusion_cell"]):
        keep.append(g if len(g) <= CAP else g.sample(CAP, random_state=SEED))
    sub = pd.concat(keep, ignore_index=True)
    print(f"\n=== 상한 {CAP} 적용 후: {len(sub)}건 ===")
    print(pd.crosstab([sub["dataset"], sub["split_role"]], sub["confusion_cell"]).to_string())
    sub.to_parquet(OUT / "cell_manifest.parquet", index=False)

    # 지시문 21절 TABLE 1
    t.reset_index().to_csv(ROOT / "results/direction_debug/cell_counts.csv", index=False)
    print(f"\n저장 -> {OUT/'cell_manifest.parquet'}, results/direction_debug/cell_counts.csv")


if __name__ == "__main__":
    main()

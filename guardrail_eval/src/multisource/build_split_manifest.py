"""향후 실험용 split manifest (지시문 16절).

원칙:
  - **무작위 행 분할만 쓰지 않는다.**
  - 같은 duplicate_group_id 와 같은 paired_group_id 는 절대 다른 split 으로 보내지 않는다.
  - 원본 split 이 있으면 존중한다.
  - source_group / canonical_dataset 을 유지해 cross-dataset 평가가 가능하게 한다.

분할 단위는 **그룹 키**다: paired_group_id 가 있으면 그것, 없으면 duplicate_group_id.
그 그룹 키를 해시해 train/val/test 로 보낸다 (source_group 안에서 층화).
"""
import hashlib, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / "data/multisource_guard", ROOT / "results/multisource_guard"
FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = "multisource_guard_v1"


def bucket(key):
    h = int(hashlib.sha1(f"{SEED}||{key}".encode()).hexdigest()[:12], 16) / 16 ** 12
    return "train" if h < FRAC["train"] else "val" if h < FRAC["train"] + FRAC["val"] else "test"


def main():
    df = pd.read_parquet(DATA / "confusion_cells.parquet")
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[
        ["sample_id", "paired_group_id", "original_split"]]
    df = df.merge(can, on="sample_id", how="left")
    # 그룹 키: 짝 그룹 > 중복 그룹
    df["group_key"] = df["paired_group_id"].fillna("").replace("", np.nan)
    df["group_key"] = df["group_key"].fillna(df["duplicate_group_id"])
    df["split"] = df["group_key"].map(bucket)

    # --- 불변식 검증 ---
    bad = df.groupby("group_key")["split"].nunique()
    assert (bad <= 1).all(), f"그룹이 split 을 넘나든다: {int((bad>1).sum())}개"
    bad2 = df.groupby("duplicate_group_id")["split"].nunique()
    n_bad2 = int((bad2 > 1).sum())
    print(f"그룹 키 {df['group_key'].nunique()}개 -> split 유일성 OK")
    print(f"duplicate_group_id 가 split 을 넘나드는 경우: {n_bad2}건 (0 이어야 함)")
    assert n_bad2 == 0

    print("\n=== split × 라벨 ===")
    print(pd.crosstab(df["split"], df["binary_main_label"]).to_string())
    print("\n=== source_group × split (TP/FP 를 만들 수 있는 출처만) ===")
    ok = df.groupby("source_group")["binary_main_label"].nunique()
    keep = ok[ok > 1].index
    sub = df[df["source_group"].isin(keep)]
    print(pd.crosstab(sub["source_group"], sub["split"]).to_string())

    cols = ["sample_id", "source_group", "canonical_dataset", "canonical_label",
            "binary_main_label", "pg2_prediction", "confusion_cell", "duplicate_group_id",
            "paired_group_id", "group_key", "original_split", "split"]
    df[cols].to_parquet(DATA / "split_manifest.parquet", index=False)
    print(f"\n저장 -> {DATA/'split_manifest.parquet'} ({len(df)}행)")


if __name__ == "__main__":
    main()

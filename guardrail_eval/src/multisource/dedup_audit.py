"""중복/겹침 감사 (지시문 8절).

중복을 **지우지 않는다.**  duplicate_group_id 를 붙여 provenance 를 보존하고,
나중에 같은 duplicate_group 이 서로 다른 split 으로 갈라지지 않게 한다.
"""
import hashlib, sys, unicodedata
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / "data/multisource_guard", ROOT / "results/multisource_guard"


def norm(t):
    return " ".join(unicodedata.normalize("NFKC", str(t)).lower().split())


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "canonical_samples.parquet")
    print(f"입력 {len(df)}행")
    df["_norm"] = df["text"].map(norm)
    df["_h"] = df["_norm"].map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:20])
    # duplicate_group_id = 정규화 텍스트 해시.  같은 해시면 같은 그룹.
    df["duplicate_group_id"] = df["_h"]

    g = df.groupby("_h")
    size = g.size()
    dup_h = size[size > 1].index
    dup = df[df["_h"].isin(dup_h)]
    print(f"정규화 기준 고유 텍스트 {len(size)}개, 중복 그룹 {len(dup_h)}개, 중복에 속한 행 {len(dup)}개")

    rows = [{"metric": "total_rows", "value": len(df)},
            {"metric": "unique_normalized_texts", "value": int(len(size))},
            {"metric": "duplicate_groups", "value": int(len(dup_h))},
            {"metric": "rows_in_duplicate_groups", "value": int(len(dup))},
            {"metric": "exact_text_duplicates",
             "value": int(len(df) - df["text"].nunique())}]

    # 서로 다른 source_group 에 걸친 중복
    cross = dup.groupby("_h")["source_group"].nunique()
    n_cross = int((cross > 1).sum())
    rows.append({"metric": "duplicate_groups_spanning_multiple_source_groups", "value": n_cross})
    # 서로 다른 canonical_dataset 에 걸친 중복
    crossd = dup.groupby("_h")["canonical_dataset"].nunique()
    rows.append({"metric": "duplicate_groups_spanning_multiple_datasets",
                 "value": int((crossd > 1).sum())})
    # 라벨이 충돌하는 중복 (같은 문장인데 SAFE/UNSAFE 둘 다)
    el = df[df["binary_main_eligible"]]
    conflict = el.groupby("_h")["binary_main_label"].nunique()
    n_conf = int((conflict > 1).sum())
    rows.append({"metric": "duplicate_groups_with_conflicting_main_label", "value": n_conf})
    print(f"  출처 그룹을 넘나드는 중복 그룹 {n_cross}개")
    print(f"  ★ 라벨이 충돌하는 중복 그룹 {n_conf}개 (같은 문장이 SAFE 이자 UNSAFE)")

    pd.DataFrame(rows).to_csv(RES / "dedup_summary.csv", index=False)

    # 출처 그룹 쌍별 겹침 상위
    pairs = []
    for h, sub in dup[dup["_h"].isin(cross[cross > 1].index)].groupby("_h"):
        gs = sorted(sub["source_group"].unique())
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                pairs.append((gs[i], gs[j]))
    if pairs:
        pc = pd.Series(pairs).value_counts().head(20)
        print("\n  출처 그룹 쌍별 중복 상위:")
        for (a, b), n in pc.items():
            print(f"    {a:42} ∩ {b:42} {n}")
        pd.DataFrame([{"group_a": a, "group_b": b, "n_shared_texts": int(n)}
                      for (a, b), n in pd.Series(pairs).value_counts().items()]
                     ).to_csv(RES / "dedup_cross_source_pairs.csv", index=False)

    df.drop(columns=["_norm", "_h"]).to_parquet(DATA / "canonical_samples.parquet", index=False)
    print(f"\n저장 -> {DATA/'canonical_samples.parquet'} (duplicate_group_id 추가), "
          f"{RES/'dedup_summary.csv'}")


if __name__ == "__main__":
    main()

"""PromptShield 을 표준 형식에 추가한다 (지시문 3B·15절).

중요한 한계 — 공개된 산출물은 `prompt` 와 `label` 두 열뿐이라 **원본 출처 표식이 없다.**
따라서 지시문 15절이 요구한 base-corpus 별 그룹
(`promptshield_alpaca` / `promptshield_dolly` / `promptshield_spp`)을 **만들 수 없다.**
대신 split 단위로 source_group 을 두고 이 한계를 명시한다.

짝(paired) 복원 시도 결과: 공격문 중 정상문과 앞 120자가 일치하는 것은 7.3% 뿐이라
train 안에서 신뢰할 만한 paired_group_id 를 복원하지 못했다 -> paired_group_id = None.
"""
import glob, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multisource.build_canonical import rec, B, PI, norm
import hashlib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/multisource_guard"
SNAP = glob.glob("/home/kana5123/.cache/huggingface/hub/datasets--hendzh--PromptShield/snapshots/*/")[0]


def main():
    rows = []
    for split in ("train", "validation", "test"):
        d = json.load(open(SNAP + f"{split}.json"))
        for i, r in enumerate(d):
            lab = int(r["label"])
            # PromptShield 는 프롬프트 주입 탐지기다.  label 1 = 주입.
            canon = PI if lab == 1 else B
            rows.append(rec(r["prompt"], "promptshield", f"promptshield:{split}",
                            f"promptshield_{split}", split, lab, canon,
                            "direct_injection" if lab == 1 else "none", "en",
                            f"{split}:{i}",
                            meta={"provenance_lost": True,
                                  "note": "released artifact has no source column"}))
    ps = pd.DataFrame(rows)
    ps["duplicate_group_id"] = ps["text"].map(
        lambda t: hashlib.sha1(norm(t).encode()).hexdigest()[:20])
    print(f"PromptShield {len(ps)}건")
    print(ps.groupby(["source_group", "binary_main_label"]).size().unstack(fill_value=0).to_string())

    old = pd.read_parquet(DATA / "canonical_samples.parquet")
    both = pd.concat([old[~old["canonical_dataset"].eq("promptshield")], ps], ignore_index=True)
    # 기존 표본과 겹치는지
    ov = len(set(ps["duplicate_group_id"]) & set(old["duplicate_group_id"]))
    print(f"기존 표본과 정규화텍스트 겹침: {ov}개 그룹")
    both.to_parquet(DATA / "canonical_samples.parquet", index=False)
    ps.to_parquet(DATA / "promptshield_only.parquet", index=False)
    print(f"저장 -> {DATA/'canonical_samples.parquet'} (총 {len(both)}행)")


if __name__ == "__main__":
    main()

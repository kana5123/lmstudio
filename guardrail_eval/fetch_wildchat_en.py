"""WildChat-4.8M 에서 영어 첫 사용자 발화만 뽑아 저장.

정상 표본용. toxic 플래그가 붙은 대화는 제외한다(실제 공격이 섞이는 것 방지).
JailbreaksOverTime 이 정상 데이터로 WildChat 을 쓴 것과 같은 취지.
"""
import json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent / "wildchat_en.jsonl"
N_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 86
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

rows, seen = [], set()
for i in range(N_SHARDS):
    f = hf_hub_download("allenai/WildChat-4.8M",
                        f"data/train-{i:05d}-of-00086.parquet", repo_type="dataset")
    df = pd.read_parquet(f, columns=["conversation", "language", "toxic", "redacted"])
    ko = df[(df["language"] == "English") & (~df["toxic"])]
    for conv in ko["conversation"]:
        u = [t for t in conv if t["role"] == "user"]
        if not u:
            continue
        t = u[0]["content"].strip()
        if t and t not in seen:
            seen.add(t); rows.append(t)
    print(f"  샤드 {i+1}/{N_SHARDS}: 누적 영어 {len(rows)}건", flush=True)
    Path(f).unlink(missing_ok=True)          # 디스크 절약 — 처리 후 삭제
    if len(rows) >= TARGET:
        break

with open(OUT, "w", encoding="utf-8") as fh:
    for t in rows:
        fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print(f"저장 {len(rows)}건 → {OUT}")

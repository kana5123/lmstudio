"""PHASE A 데이터 감사 실행: 레지스트리 / 길이 / 중복 / base 추론 / 혼동셀."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_verifier.config import DATA, DOCS, RES, load_runtime, probe_positive_label
from src.decompx_verifier.data_audit import (build_registry, load_canonical, load_jot, norm_text,
                                             run_base_inference)

model, tok, rc = load_runtime()
pos_id, pa, pb = probe_positive_label(model, tok)
rc.positive_label_id = pos_id
print(f"양성 라벨 id={pos_id} (실측: 공격 {[round(x,4) for x in pa]}, 정상 {[round(x,4) for x in pb]})")
print(f"층 {rc.num_hidden_layers}, 은닉 {rc.hidden_size}, 최대길이 {rc.max_position_embeddings}, "
      f"transition {rc.n_transitions}\n")

can, jot = load_canonical(), load_jot()
reg = build_registry(can)
RES.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
reg.to_csv(RES / "dataset_registry.csv", index=False)
print("=== §2 라벨 호환성 ===")
print(reg.groupby(["use", "original_label"]).n.sum().to_string(), "\n")

# JOT 은 canonical 과 컬럼을 맞춰 이어붙인다
jot["duplicate_group_id"] = jot.group_id
allrows = pd.concat([can, jot], ignore_index=True)
allrows = allrows[allrows.use != "EXCLUDE"].reset_index(drop=True)

# §5 중복: 기존 duplicate_group_id 를 재사용, 없으면 정규화 텍스트 해시
miss = allrows.group_id.isna()
if miss.any():
    import hashlib
    allrows.loc[miss, "group_id"] = [hashlib.blake2b(norm_text(t).encode(),
                                                     digest_size=10).hexdigest()
                                     for t in allrows.text[miss]]
print(f"§5 그룹: 표본 {len(allrows):,} -> 고유 그룹 {allrows.group_id.nunique():,} "
      f"(기존 id 재사용 {int((~miss).sum()):,}, 새 해시 {int(miss.sum()):,})\n")

pred = run_base_inference(allrows, model, tok, rc, pos_id)
pred.to_parquet(DATA / "pg2_predictions.parquet", index=False)

# §4 길이 초과 보고
over = pred[~pred.length_ok]
over[["dataset", "sample_id", "token_length"]].to_csv(RES / "length_exclusions.csv", index=False)
print(f"=== §4 길이 정책 (최대 {rc.max_position_embeddings} 토큰, 자르지 않음) ===")
print(f"초과로 MAIN 제외: {len(over):,} / {len(pred):,} ({len(over)/len(pred)*100:.2f}%)")
if len(over):
    print(over.groupby("dataset").size().sort_values(ascending=False).head(8).to_string())
print()

# §6 혼동셀
ok = pred[pred.length_ok]
cnt = (ok.groupby(["use", "dataset"]).confusion_cell.value_counts().unstack(fill_value=0)
       .reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0).reset_index())
cnt["n"] = cnt[["TP", "FP", "TN", "FN"]].sum(1)
cnt["wrong"] = cnt.FP + cnt.FN
cnt.to_csv(RES / "confusion_counts.csv", index=False)
print("=== §6 혼동셀 (길이 통과분, native argmax) ===")
print(cnt.sort_values(["use", "dataset"]).to_string(index=False), "\n")
m = cnt[cnt.use == "MAIN"][["TP", "FP", "TN", "FN", "n", "wrong"]].sum()
print(f"MAIN 합계: TP {m.TP:,}  FP {m.FP:,}  TN {m.TN:,}  FN {m.FN:,}  "
      f"(오답 {m.wrong:,} = {m.wrong/m.n*100:.1f}%)")
print(f"저장 -> {RES}/dataset_registry.csv, confusion_counts.csv, length_exclusions.csv")
print(f"       {DATA}/pg2_predictions.parquet")

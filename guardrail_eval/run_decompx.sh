#!/bin/bash
# DecompX 특징 추출 병렬 실행: GPU 5,6,7,8 에 2개씩 = 8 샤드
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
NS=8
for SPLIT in ver_train ver_dev eval_val eval_test; do
  for S in $(seq 0 $((NS-1))); do
    GPU=$((5 + S % 4))
    CUDA_VISIBLE_DEVICES=$GPU $PY src/features/extract_decompx_features.py \
      --split $SPLIT --shard $S --nshards $NS > logs/dx_${SPLIT}_$S.log 2>&1 &
  done
  wait
  echo "=== $SPLIT 완료 ==="
done
echo ALL_DONE

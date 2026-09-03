#!/bin/bash
# 방향성 토큰 기여 이동(a) 추출.  held-out 을 먼저 돌려 답을 빨리 얻는다.
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
NS=8
for SPLIT in eval_test eval_val ver_dev ver_train; do
  for S in $(seq 0 $((NS-1))); do
    GPU=$((5 + S % 4))
    CUDA_VISIBLE_DEVICES=$GPU $PY src/directional/extract_directional.py \
      --split $SPLIT --shard $S --nshards $NS > logs/dir_${SPLIT}_$S.log 2>&1 &
  done
  wait
  echo "=== $SPLIT 완료 $(date +%H:%M) ==="
done
echo DIR_ALL_DONE

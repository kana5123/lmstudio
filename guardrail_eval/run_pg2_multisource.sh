#!/bin/bash
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
for S in 0 1 2 3; do
  GPU=$([ $((S%2)) -eq 0 ] && echo 5 || echo 9)
  CUDA_VISIBLE_DEVICES=$GPU $PY src/multisource/pg2_inference.py --shard $S --nshards 4 \
    > logs/ms_pg2_$S.log 2>&1 &
done
wait
echo MS_PG2_DONE

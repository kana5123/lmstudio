#!/bin/bash
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
for S in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((5+S)) $PY src/direction_debug/extract_cell_hidden.py \
    --shard $S --nshards 5 > logs/dd_h_$S.log 2>&1 &
done
wait
echo CELL_HIDDEN_DONE

#!/bin/bash
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((5+S)) $PY src/direction_repro/extract_hidden.py \
    --group wildjailbreak:adversarial --shard $S --nshards 4 > logs/rp_h_$S.log 2>&1 &
done
wait
echo REPRO_HIDDEN_DONE

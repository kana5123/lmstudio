#!/bin/bash
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
CUDA_VISIBLE_DEVICES=5 $PY src/direction_debug/extract_cell_hidden.py --shard 2 --nshards 5 > logs/dd_h_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY src/direction_debug/extract_cell_hidden.py --shard 3 --nshards 5 > logs/dd_h_3.log 2>&1 &
wait
echo CELL_HIDDEN_FIX_DONE

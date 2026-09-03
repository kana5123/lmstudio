#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for b in piarena_ignore piarena_completion piarena_character piarena_combined; do
  for g in piguard llamafirewall protectaiv2; do
    echo "=== $b / $g $(date +%H:%M:%S)"
    if ../.venv/bin/python rfpr.py $g $b > logs/rfpr_${b}_${g}.log 2>&1; then
      grep -v "^Error preprocessing" logs/rfpr_${b}_${g}.log | tail -1 | cut -c1-160
    else echo "!!! 실패"; tail -4 logs/rfpr_${b}_${g}.log; exit 1; fi
  done
done
echo "=== DONE $(date +%H:%M:%S)"

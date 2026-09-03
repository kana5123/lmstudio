#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in piguard llamafirewall protectaiv2; do
  echo "=== combined / $g $(date +%H:%M:%S)"
  if ../.venv/bin/python rfpr.py $g piarena_combined > logs/rfpr_piarena_combined_${g}.log 2>&1; then
    grep -v "^Error preprocessing" logs/rfpr_piarena_combined_${g}.log | tail -1 | cut -c1-170
  else echo "!!! 실패"; tail -4 logs/rfpr_piarena_combined_${g}.log; exit 1; fi
done
echo "=== DONE $(date +%H:%M:%S)"

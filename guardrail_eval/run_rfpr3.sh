#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for b in piarena jailbreak; do
  for g in deepset fmops protectaiv2 promptguard_v1 promptguard_v1_strict piguard; do
    echo "=== $b / $g $(date +%H:%M:%S)"
    if ../.venv/bin/python rfpr.py $g $b > logs/rfpr_${b}_${g}.log 2>&1; then
      tail -1 logs/rfpr_${b}_${g}.log
    else echo "!!! 실패"; tail -6 logs/rfpr_${b}_${g}.log; exit 1; fi
  done
done
echo "=== DONE $(date +%H:%M:%S)"

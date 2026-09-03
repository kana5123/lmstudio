#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-7}
for b in piarena jailbreak; do
  for g in deepset fmops protectaiv2 promptguard_v1 promptguard_v1_strict piguard llamafirewall qwen3guard; do
    echo "=== $b / $g $(date +%H:%M:%S)"
    if ../.venv/bin/python rfpr.py $g $b > logs/rfpr_${b}_${g}.log 2>&1; then
      grep -v "^Error preprocessing" logs/rfpr_${b}_${g}.log | tail -1
    else echo "!!! 실패"; tail -5 logs/rfpr_${b}_${g}.log; exit 1; fi
  done
done
echo "=== DONE $(date +%H:%M:%S)"

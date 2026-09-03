#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in qwen3guard sguard; do
  echo "=== multijail_ko / $g $(date +%H:%M:%S)"
  if ../.venv/bin/python rfpr.py $g multijail_ko > logs/rfpr_multijail_ko_${g}.log 2>&1; then
    echo "   ok"
  else echo "!!! 실패"; tail -5 logs/rfpr_multijail_ko_${g}.log; exit 1; fi
done
echo "=== DONE $(date +%H:%M:%S)"

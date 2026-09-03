#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for b in piarena jailbreak; do
  for g in llamafirewall qwen3guard nemoguard; do
    echo "=== $b / $g  시작 $(date +%H:%M:%S)"
    if ../.venv/bin/python rfpr.py $g $b > logs/rfpr_${b}_${g}.log 2>&1; then
      grep -v "^Error preprocessing" logs/rfpr_${b}_${g}.log | tail -1
    else
      echo "!!! 실패"; tail -6 logs/rfpr_${b}_${g}.log; exit 1
    fi
  done
done
echo "=== ALL DONE $(date +%H:%M:%S)"

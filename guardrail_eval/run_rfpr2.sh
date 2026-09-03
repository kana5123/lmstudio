#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in qwen3guard nemoguard; do
  echo "=== jailbreak / $g 시작 $(date +%H:%M:%S)"
  if ../.venv/bin/python rfpr.py $g jailbreak > logs/rfpr_jailbreak_${g}.log 2>&1; then
    tail -1 logs/rfpr_jailbreak_${g}.log
  else echo "!!! 실패"; tail -6 logs/rfpr_jailbreak_${g}.log; exit 1; fi
done
echo "=== DONE $(date +%H:%M:%S)"

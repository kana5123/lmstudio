#!/bin/bash
# 모델별 독립 프로세스. 하나가 죽어도 나머지는 계속 진행.
cd /home/kana5123/ETRI/guardrail_eval
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=${GPU:-9}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in madlad3b llama3.1-8b qwen2.5-7b qwen3-8b gemma2-9b gemma3-12b exaone3.5-7.8b kanana1.5-8b mistral-7b; do
  echo "=== $g 시작 $(date +%H:%M:%S)"
  timeout 3600 ../.venv/bin/python mt_bakeoff.py "$g" 2>&1 \
    | grep -vE "Loading|warnings|You passed|Fetching|Downloading|^ *$" | tail -3
  echo "   종료코드 $? ($(date +%H:%M:%S))"
done
echo "=== ALL DONE $(date +%H:%M:%S)"

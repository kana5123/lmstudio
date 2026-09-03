#!/bin/bash
set -u
cd /home/kana5123/ETRI/guardrail_eval
# 앞선 실행(seq2seq 5종)이 끝날 때까지 대기
while pgrep -f "[m]t_bakeoff.py" > /dev/null; do sleep 30; done
echo "=== 1차(seq2seq) 종료, LLM 배치 시작 $(date +%H:%M:%S)"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../.venv/bin/python mt_bakeoff.py 2>&1 | grep -vE "Loading|warnings|You passed|Fetching|^ *$"
echo "=== DONE $(date +%H:%M:%S)"

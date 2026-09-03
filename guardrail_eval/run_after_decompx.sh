#!/bin/bash
# DecompX 추출이 끝나는 즉시 남은 단계를 이어서 실행한다.
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
until grep -q ALL_DONE logs/run_decompx.log 2>/dev/null; do sleep 30; done
echo "=== DecompX 완료, 후속 단계 시작 $(date +%H:%M) ==="

echo "== 토큰 기여 분석 =="
$PY src/analysis/analyze_token_contributions.py --split eval_test > logs/token_contrib.log 2>&1
tail -25 logs/token_contrib.log

echo "== 절제 실험 전체 (순서 고정 수정 반영해 처음부터 다시) =="
CUDA_VISIBLE_DEVICES=5 $PY src/train/train_verifier.py 2>&1 | tee logs/ablations.log | grep -E "params=|B0"

echo "== 증류 (B9) =="
CUDA_VISIBLE_DEVICES=6 $PY src/train/distill_decompx.py 2>&1 | tee logs/distill.log | grep -E "상관|AUROC"

echo "== 최종 평가 =="
$PY src/eval/evaluate_verifier.py 2>&1 | tee logs/verifier_table.log | tail -20
$PY src/eval/evaluate_final_guard.py 2>&1 | tee logs/final_guard.log | tail -30

echo "== 지연시간 =="
CUDA_VISIBLE_DEVICES=5 $PY src/eval/benchmark_latency.py --bs 1 --with_mdeberta 2>&1 | tee logs/latency_gpu.log | tail -8

echo "== 그림 =="
$PY src/analysis/plot_results.py
echo "PIPELINE_DONE $(date +%H:%M)"

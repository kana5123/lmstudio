#!/bin/bash
# 전체 파이프라인 재현 스크립트.  순서가 의존관계다.
set -e
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
G=${G:-9}                       # 단일 GPU 실행용

echo "== 0. 포팅 정확성 테스트 =="
CUDA_VISIBLE_DEVICES=$G $PY tests/test_logits_equivalence.py
CUDA_VISIBLE_DEVICES=$G $PY tests/test_decomposition_reconstruction.py
$PY tests/test_split_leakage.py

echo "== 1. 분할 + PG2 동결 추론 =="
$PY src/data/splits.py
CUDA_VISIBLE_DEVICES=$G $PY src/data/build_verifier_dataset.py

echo "== 2. 층별 CLS 은닉표현 =="
CUDA_VISIBLE_DEVICES=$G $PY src/features/extract_pg2_hidden.py

echo "== 3. TP/FP 방향 (ver_train 전용) =="
$PY src/features/build_tp_fp_direction.py

echo "== 4. 1단계 가설 + 교란요인 대조 =="
$PY src/analysis/analyze_cls_shift.py
$PY src/analysis/confound_check.py

echo "== 5. DecompX 특징 (병렬, GPU 5-8) =="
./run_decompx.sh

echo "== 6. 토큰별 은닉표현 (증류 학생 입력) =="
for S in ver_train ver_dev eval_val eval_test; do
  CUDA_VISIBLE_DEVICES=$G $PY src/features/extract_token_hidden.py --split $S
done

echo "== 7. 토큰 기여 분석 =="
$PY src/analysis/analyze_token_contributions.py --split eval_test

echo "== 8. 절제 실험 (B1~B7 + 용량대조) =="
CUDA_VISIBLE_DEVICES=$G $PY src/train/train_verifier.py

echo "== 9. mDeBERTa 텍스트 기준선 (B8) =="
CUDA_VISIBLE_DEVICES=$G $PY src/models/mdeberta_verifier.py

echo "== 10. DecompX 증류 (B9) =="
CUDA_VISIBLE_DEVICES=$G $PY src/train/distill_decompx.py

echo "== 11. 최종 평가 =="
$PY src/eval/evaluate_verifier.py
$PY src/eval/evaluate_final_guard.py

echo "== 12. 지연시간/메모리 (전용 GPU 필요) =="
CUDA_VISIBLE_DEVICES=$G $PY src/eval/benchmark_latency.py --bs 1 --with_mdeberta
$PY src/eval/benchmark_latency.py --device cpu --bs 1

echo ALL_DONE

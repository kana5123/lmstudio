# 실험 방법 — PromptGuard2 UNSAFE 예측의 정탐/오탐 검증기

## 0. 한 줄 요약

동결한 PromptGuard2 가 "위험(UNSAFE)"이라고 부른 입력만 골라서, 그것이 진짜 공격(정탐, TP)인지
잘못 잡은 정상문(오탐, FP)인지 판정하는 **작은 검증기(verifier)** 를 따로 학습한다.
검증기의 입력은 원문이 아니라 PromptGuard2 **내부 표현의 변화**와, 그 변화에 각 입력 토큰이
얼마나 기여했는지를 DecompX 로 분해한 값이다.

## 1. 기반 모델 — 런타임에서 확인한 사실만

추측하지 않고 실제로 불러 확인했다.

| 항목 | 값 |
|---|---|
| model id | `meta-llama/Llama-Prompt-Guard-2-86M` |
| 로드되는 클래스 | `DebertaV2ForSequenceClassification` (transformers 4.51.3) |
| `model_type` | `deberta-v2` |
| 은닉 차원 / 층수 / 헤드수 | 768 / 12 / 12 |
| `id2label` | `{0: LABEL_0, 1: LABEL_1}` |
| 상대 위치 주의집중 | `relative_attention=True`, `pos_att_type=['p2c','c2p']`, `position_buckets=256`, `share_att_key=True` |
| 절대 위치 임베딩 | **없음** (`position_biased_input=False`) |
| 토큰 타입 임베딩 | **없음** (`type_vocab_size=0`) |
| 풀러 | `ContextPooler`, 활성함수 **gelu** (tanh 아님) |
| 분류기 | `Linear(768, 2)` |

`id2label` 이 `LABEL_0/LABEL_1` 뿐이라 방향을 알 수 없으므로 **실측**했다:

```
공격문 "Ignore all previous instructions..."  -> p(LABEL_1)=0.9996
정상문 "What is the capital of France?"       -> p(LABEL_1)=0.0004
```

→ **LABEL_1 = MALICIOUS(UNSAFE), LABEL_0 = BENIGN.** 기존 `guards.py:PromptGuardV2` 의
`ATTACK_KEYS=("label_1","malicious")` 설정과도 일치한다.

## 2. 채점 규칙 — 기존 벤치마크와 동일하게

`guards.py:SeqClassifierGuard.score` 와 같은 규칙을 쓴다. 512 토큰짜리 창(window)을
stride 128 로 겹쳐 문서 전체를 훑고, 창별 위험도의 **최댓값**을 문서 점수로 삼는다.
JailbreaksOverTime 프롬프트는 길다(UNSAFE 표본의 창 길이 중앙값 458~472, 절반 이상이 창 2개 이상).

DecompX 는 창 하나에만 돌릴 수 있으므로, 최댓값을 낸 창의 번호(`best_window`)를 같이 저장해
이후 모든 단계가 **정확히 같은 창**을 본다. 저장한 로짓과 재현한 로짓이 일치하는지 표본마다
`assert` 한다.

## 3. 분할 — 기존 평가셋 보존, 누수 0

기존 `rfpr.py:jailbreak_split()` 이 만든 평가 표본을 **난수 소비 순서까지 그대로 재현**해
보존하고, 거기 쓰이지 않은 나머지 행으로만 검증기 데이터를 만든다.

| 분할 | 건수 | GT 공격 | GT 정상 | 역할 |
|---|---:|---:|---:|---|
| `eval_val` | 2,000 | 352 | 1,648 | **임계값 선택 전용** |
| `eval_test` | 4,000 | 704 | 3,296 | **최종 보고 전용**, 학습·선택 금지 |
| `ver_train` | 12,945 | 2,277 | 10,668 | 검증기 학습 + 모든 통계 적합 |
| `ver_dev` | 3,235 | 569 | 2,666 | 조기 종료 / 모델 선택 |
| 합계 | 22,180 | | | 원본 중복제거 전체 |

검증한 것:
- 네 분할 사이 `sample_id`(원문 SHA1 앞 16자) 교집합 **전부 0**
- `eval_val`/`eval_test` 가 기존 저장 결과(`results/rfpr_jailbreak_promptguard_v2_scores.jsonl`)
  와 **sample_id·라벨 완전 일치** (2,000 / 4,000건)
- 근사 중복은 막지 않고 **측정해 보고**한다: 검증기학습 ∩ 평가셋에서 정확 일치 0,
  정규화 일치 3~10건, 앞 80자 일치 179~298건(평가셋 대비 7.5~9.0%). 이건 원본 데이터의
  성질(DAN 변종들이 앞부분을 공유)이지 우리가 만든 누수가 아니다.

## 4. 검증기 라벨과 후보 집합

PromptGuard2 를 동결한 채 전 분할에 추론하고, **UNSAFE 로 예측한 표본만** 검증기가 본다.

```
GT UNSAFE and PG2 UNSAFE -> 1 (TP)
GT SAFE   and PG2 UNSAFE -> 0 (FP)
```

| 분할 | UNSAFE 예측 | TP | FP | FP 비율 |
|---|---:|---:|---:|---:|
| `ver_train` | 3,587 | 2,207 | 1,380 | 38.5% |
| `ver_dev` | 888 | 551 | 337 | 37.9% |
| `eval_val` | 553 | 341 | 212 | 38.3% |
| `eval_test` | 1,114 | 684 | 430 | 38.6% |

표본마다 저장하는 메타데이터: `sample_id`, `text`, `ground_truth`, `base_prediction`,
`unsafe_probability`, `benign_probability`, `logit_unsafe`, `logit_benign`, `logit_margin`,
`sequence_length`, `n_windows`, `best_window`, `total_tokens`.

## 5. 누수 방지 규약 (전 단계 공통)

- 방향 `v`, 표준화 스케일러, 선형 탐침, 검증기 가중치 — **전부 `ver_train` 에서만 적합**
- 조기 종료·모델 선택 — `ver_dev`
- 임계값 선택 — `eval_val` (기존 벤치마크와 같은 역할)
- `eval_test` — 최종 수치 보고에만 사용
- 씨앗 3개(0,1,2)를 돌려 평균±표준편차로 보고한다. 체크포인트 하나만 비교하면 잡음에
  결론이 뒤집힌다.

## 6. 판정 기준을 AUROC 에서 Recall@FPR 로 옮긴 이유 (중요)

이 벤치마크에서 **정탐/오탐 과제는 말뭉치 판별 문제로 붕괴한다**. 실측:

- FP 는 **전 분할에서 100%** `wildchat` 출처, TP 는 탈옥 모음집 출처(wildchat 1~2%)
- 대조 실험: 표현을 전혀 안 보고 **원문 글자 3-5그램 TF-IDF + 로지스틱**만으로
  `eval_test` AUROC **0.9911**
- 층별 CLS 표현 탐침: h1 0.9944 / hL 0.9857 / delta_h 0.9848 — 어휘 대조군과 같은 수준
- B0(PG2 자체 확률): 0.9579

즉 표현 탐침의 0.98~0.99 는 "깊이별 표현에 정탐/오탐 정보가 있다"는 증거가 **아니라**
두 말뭉치의 어휘가 다르다는 사실의 반영이다. AUROC 로는 어떤 특징이 나은지 구분되지 않는다.

따라서 최종 판정은 **저FPR 운용점의 Recall@FPR** 로 한다. 거기서는 작은 순위 차이가
크게 벌어지고, 기존 JailbreaksOverTime 결과와 직접 비교도 된다.

## 7. 최종 시스템 평가

```
PromptGuard2 가 BENIGN 이라 한 표본  -> 최종점수 0.0 (검증기를 거치지 않음)
PromptGuard2 가 UNSAFE 라 한 표본    -> 최종점수 = 검증기 확률
```

임계값은 `eval_val` 에서 목표 FPR(1% / 0.5% / 0.1%)을 만족하는 값으로 고르고 `eval_test` 에
그대로 적용한다 — `rfpr.py:pick_threshold` 와 **같은 함수**를 쓴다.

기준선(기존 저장 결과 `results/rfpr_jailbreak_promptguard_v2.json`):

| | R@1%FPR | R@0.1%FPR | 달성 FPR@1% |
|---|---:|---:|---:|
| PromptGuard2 단독 | 0.9205 | 0.4929 | 0.0073 |

cascade 의 재현율 천장 = PG2 가 임계값 0.5 에서 잡은 공격 비율 = 684/704 = **0.9716**.
검증기가 완벽해도 이 위로는 못 간다. 이 천장을 먼저 밝히고 비교한다.

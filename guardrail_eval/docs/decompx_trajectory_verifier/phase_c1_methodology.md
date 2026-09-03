# PHASE C1 방법론

regime: **native_argmax_feasibility**.  최종 low-FPR deployment 실험이 아니다.

## 대상
PromptGuard2 가 ATTACK 이라 예측한 표본만 검증한다.  target: TP=0, FP=1.
출력은 `P(FP | PromptGuard2 predicted ATTACK)`.

## 동결
아키텍처는 PHASE B2 에서 검증한 A3 를 그대로 쓴다.
`artifacts/decompx_trajectory_verifier/frozen_architecture_config.json`
(총 1,004,417 파라미터, source_tree_hash `899bff5a6c5d196f97e8c8bde47bc7d4`).

## 학습한 세 모델 (§10)
| 모델 | 입력 | 파라미터 |
|---|---|---:|
| M0 | `[z_benign, z_attack, z_attack-z_benign]` -> 3-16-1 | 81 |
| A0 | 토큰별 `[Y_B, Y_A, a]` 만 (중간 C 궤적 없음) | 491,393 |
| A3 | 전 층 C 궤적 + 최종 Y/a 앵커 | 1,004,417 |

동일 split / seed / 표본 / 조기종료 규칙을 쓴다.  A3 의 용량이 더 큰 점은 기록만 하고
이번 단계에서 capacity matching 을 추가하지 않는다(§11).

## 분할
duplicate_group_id 단위, (source_group, confusion_cell) 층별 배정.
seen-source 70/15/15, LOSO 는 학습 source 내부에서 82.4/17.6 + held-out source 전체가 test.
held-out source 라벨은 학습·조기종료·하이퍼파라미터·정규화 어디에도 쓰지 않는다.

## 누수 차단 (§5)
dataset / source_group / source_subgroup / original_split / confusion_cell /
duplicate_group_id / sample_id 는 verifier 입력에 넣지 않는다.  분할·표집·평가·감사에만 쓴다.
raw text, TF-IDF, 토큰 빈도는 이번 PHASE 에서 쓰지 않는다.

## a 재계산 (§3)
`a` 는 캐시값을 쓰지 않고 매 표본 `Y[:,attack] - Y[:,benign]` 으로 다시 계산하며
캐시값과 대조한다.  전 60 run 최대 편차 **0.0**.

## 조기종료 (§13)
source 별 validation AUPRC 의 macro 평균.  큰 WildJailbreak 가 지배하지 않게 한다.
LOSO 는 학습에 포함된 source 들의 macro 평균만 쓴다.

## I/O (§18)
92.5 GB fp32 캐시를 numpy memmap 으로 스트리밍한다(전체를 RAM 에 올리지 않음).
토큰 길이 버킷팅으로 패딩을 줄인다.  A3 학습에서 데이터 대기 비율 0.025~0.054.
I/O 때문에 아키텍처를 바꾸지 않았다.

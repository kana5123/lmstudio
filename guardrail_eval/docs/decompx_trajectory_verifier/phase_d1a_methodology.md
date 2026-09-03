# PHASE D1A 방법론 — LAYER-SUBSET RETRAINING AND LENGTH-CONTROLLED TRANSFER

D0 의 층 occlusion 은 L1~L12 전체로 학습된 A3 에서 **추론 시점에** 특정 층을 0 으로 만든
진단이었다.  D1A 는 사용할 DecompX 층 범위를 **처음부터 제한하고 초기화부터 새로 학습**해
후기 층이 OOD 실패의 실제 원인인지 검증한다.

## 학습한 다섯 variant
| variant | 이름 | 층 범위 | L_eff | 총 파라미터 |
|---|---|---|---:|---:|
| V0 | A0 FINAL ATTRIBUTION ONLY | 없음 (Y/a 만) | — | 491,393 |
| V1 | A3-FULL | L1–L12 | 12 | 1,004,417 |
| V2 | A3-EARLY | L1–L8 | 8 | 1,003,905 |
| V3 | A3-LATE | L9–L12 | 4 | 1,003,393 |
| V4 | A3-FINAL | L12 | 1 | 1,003,009 |

V1~V4 에서 달라지는 것은 **depth positional embedding 크기와 depth sequence 길이뿐**이다
(CellProjector 99,968 → 98,560).  CellProjector / AttributionAnchor / Fusion /
TokenContextEncoder / Head 는 완전히 동일하다.  V1 은 기존 C1 의 A3 와
동일 가중치에서 **출력 최대차 0.0** 으로 일치함을 확인했다.
V0 는 아키텍처가 다르므로 별도 기준선으로 명시한다.

## 동결한 것
PHASE C1 의 split manifest 를 그대로 재사용했다(sample_id / duplicate_group / test 표본 /
seed 별 manifest 무변경).  학습 설정도 C1 과 같다: AdamW, lr 1e-4, wd 1e-2, max 30 epoch,
patience 5, grad clip 1.0, seed 0~4.  조기종료는 학습에 포함된 source 들의 validation
AUPRC macro 평균이며, held-out source 결과로 checkpoint 를 고르지 않았다.
sampler 도 C1 그대로(source_group 균형 → source 내부 TP/FP 균형)이고,
token length 를 verifier 입력 feature 로 넣지 않았다.

## 길이 매칭
32 토큰 폭 16 구간에서 각 (source, split, 구간) 마다 `n_match = min(num_TP, num_FP)` 만큼
TP/FP 를 같은 수로 뽑았다.  선택은 `hash(sample_id, seed)` 순서로 결정론적이며 한 matched set
안에서 표본을 중복 사용하지 않는다.  **matched 집합에 맞춰 재학습하지 않고**, 자연 분포에서
학습한 동일 checkpoint 를 그대로 평가했다.  matched 집합의 AUPRC 기준선은 0.5 다.
test 결과를 보고 매칭을 조정하지 않았다.

## 개발 데이터 상태
WildJailbreak / PromptShield / Question Set 은 이미 architecture 개발, C1 평가, D0 진단,
그리고 이번 D1A 에 사용됐다.  **이후 논문에서 이 셋을 "untouched external test" 라고 부를 수
없다.**  최종 OOD 주장은 완전히 새로운 benchmark 에서 architecture 를 freeze 한 뒤 검증해야 한다.

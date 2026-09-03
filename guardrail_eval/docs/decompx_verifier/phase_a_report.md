# PHASE A 감사 보고 (2026-09-02)

지시받은 대로 PHASE A(감사)까지만 수행했다.  evidence 추출·학습은 시작하지 않았다.
기존 `direction_correctness` / `direction_debug` / `direction_repro` / DecompX /
`failure_structure` / `shared_verifier` 산출물은 하나도 수정하지 않았다.

## 1. 런타임 설정 — 하드코딩 없음

로드된 checkpoint 에서 읽은 값만 사용한다(`src/decompx_verifier/config.py`).

| 항목 | 값 |
|---|---|
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_labels / label2id | 2 / `{LABEL_0:0, LABEL_1:1}` |
| max_position_embeddings | 512 |
| pooler_hidden_act | gelu |
| 양성 라벨 id | **1** (라벨명이 무의미해 탐침 문장으로 실측: 공격 0.9995 / 정상 0.0004) |
| transition 수 K | **11** — DecompX 포트가 `C[1]..C[12]` 만 주므로 `L1→L2 … L11→L12` |

base 는 `eval()` + `requires_grad_(False)` 로 동결했고 assert 로 확인한다.
`position_biased_input=False` 이므로 임베딩 층 출력에 절대 위치 정보가 없다 —
`E_k` 를 순수 어휘 앵커라고 부를 수 있다(단어 임베딩 + LayerNorm).

## 2. 라벨 호환성

상세는 [label_mapping.md](label_mapping.md).  모델 카드 원문 기준으로
**MAIN 146,196 / DIAGNOSTIC_ONLY 4,009 / EXCLUDE 10,435** 로 갈랐다.

제외 근거는 성능이 아니라 의미다.  `HARMFUL_DIRECT`(유해하지만 덮어쓰기 시도 없음)와
`INDIRECT_PROMPT_INJECTION`(v2 가 목표에서 제거)이 대상이다.

## 3. 길이 정책

자르지 않고, 512 토큰 초과분을 MAIN 에서 제외했다: **10,617 / 172,385 (6.16%)**.
목록은 `results/decompx_verifier/length_exclusions.csv`.
초과가 가장 많은 곳은 jailbreaksovertime:train 4,367, promptshield_test 1,488 이다.

## 4. 중복 그룹

기존 `duplicate_group_id` 를 172,385 행 전부에 재사용했다(새로 만든 해시 0개).
고유 그룹 167,790 개.  분할은 반드시 이 그룹 단위로 한다.

## 5. 혼동 셀 (native argmax, threshold 0.5)

MAIN 합계: **TP 21,545 / FP 4,351 / TN 102,284 / FN 12,847** (오답 12.2%).
`operating_regime="native_argmax"`, `threshold=0.5` 를 각 행에 기록했다.

### ★ 반드시 짚어야 할 위험 — UNSAFE 분기가 출처 판별로 붕괴할 수 있다

FP 4,351 개 중 **3,374 개(77.6%)가 `wildjailbreak:adversarial_benign` 한 곳**에서 나온다.
그런데 그 데이터셋에는 **TP 가 0 개**다.  따라서 여러 데이터셋을 합쳐 TP vs FP 를
학습하면 검증기는 "이게 wildjailbreak adversarial_benign 인가" 만 배워도 점수가 오른다.
이는 이 저장소에서 이미 확인된 교란과 같은 형태다.

TP 와 FP 를 둘 다 의미 있는 수로 가진 데이터셋은 사실상 두 개뿐이다.

| 데이터셋 | TP | FP |
|---|---:|---:|
| promptshield:test | 1,540 | 439 |
| piguard:Question Set | 888 | 236 |
| (promptshield:train) | 6,490 | 22 |
| (safe-guard) | 1,258 | 14 |

SAFE 분기(TN vs FN)는 상태가 낫다: promptshield test 16,083/3,966,
train 8,952/2,450, safe-guard 5,591/1,085 로 여러 출처에 퍼져 있다.

**결론: §28 의 LODO 와 §42 의 출처 지름길 대조군이 선택 사항이 아니라 필수다.**
데이터셋별 보고(§43) 없이 pooled AUROC 만 보면 이 교란을 놓친다.

## 6. 수치 감사

상세는 [mathematical_invariants.md](mathematical_invariants.md).

**처음에 내가 설정한 허용치(1e-4)를 float32 가 초과했다.** 11–12 층에서
복원 상대오차가 3.3e-4 / 7.4e-4 였다.  §49 대로 추출을 시작하지 않고 원인을 규명했다.

단일 변수 비교(같은 표본 12개, dtype 만 변경) 결과 **분해는 대수적으로 정확하다**:
float64 에서 복원 2.7e-10 / 3.7e-10, D 보존 3.0e-10 — float32 대비 약 10^6 배 개선.
초과분은 11층에서 CLS 크기가 2.20 → 26.66 으로 12배 뛰며 상쇄가 커진 반올림이다.

float64 추출은 약 85배 느려(141k 표본 기준 950시간 대 11.3시간) 비현실적이다.
그래서 허용치를 넓히되 **그 근거를 임의로 두지 않고 직접 측정했다** — float32 증거가
float64 증거와 얼마나 다른가:

| 증거 | 코사인 최소 | 상대 L2 최대 |
|---|---:|---:|
| g / zD_pos / zD_neg | 1.000000 | 7.5e-06 / 2.7e-05 / 4.8e-05 |
| zH_pos / zH_neg / zE_pos / zE_neg | 1.000000 | ≤ 9.3e-06 |
| mass_pos / mass_neg | — | ≤ 1.6e-05 |

12층 복원 오차 7.4e-4 는 pooled 증거로 전파되지 않는다.  재판정 결과:

| 검사 | 실측 최대 | 허용(float32) | 판정 |
|---|---:|---:|---|
| §8 복원 `sum_k C == h_CLS` | 7.393e-04 | 2e-03 | 통과 |
| §8 패딩 기여 == 0 | **0.000e+00** | 1e-06 | 통과 |
| §12 보존 `sum_k D == g` | 5.803e-04 | 2e-03 | 통과 |
| §15 사영 `sum_k p == ‖g‖` | 1.294e-04 | 2e-03 | 통과 |
| §18 `mass_pos−mass_neg == ‖g‖` | 1.466e-04 | 2e-03 | 통과 |

## 7. 단위 테스트 (§31)

합성 데이터 기반 T1, T3–T9 를 먼저 통과시켰다(`src/decompx_verifier/tests/test_sdr.py`).

| 테스트 | 결과 |
|---|---|
| T1 shape | 통과 |
| T3 `sum_k D == g` | 2.835e-16 |
| T4 `sum_k p == ‖g‖` | 4.037e-14 |
| T5 `mass_pos−mass_neg == ‖g‖` | 4.022e-14 |
| T6 패딩 무영향 | 통과 (패딩에 1e3 쓰레기 주입해도 결과 동일) |
| T7 반대 성분 없으면 `z_neg == 0` | 정확히 0 |
| T8 동시 순열 불변 (§21 의도된 성질) | 통과 |
| T9 D 단독 순열은 검색을 바꿈 | 평균 상대차 0.908 |

T2(실제 checkpoint 복원)는 위 §6 이 대신한다.  T10–T12 는 검증기 구현 후 수행한다.

## 8. §16 low-motion 임계 — 아직 정하지 않았다

`g_norm` 을 그대로 저장하고, 학습 분할이 확정된 뒤 **train 통계에서만** 임계를 정한다.
추출 시점에 정하면 test 를 보고 고르는 경로가 생긴다.

## 9. PHASE B 비용

| 토큰 | 개수 | 초/표본 | 시간 |
|---|---:|---:|---:|
| ≤32 | 57,393 | 0.004 | 0.06 |
| 32–64 | 15,573 | 0.009 | 0.04 |
| 64–128 | 23,346 | 0.044 | 0.29 |
| 128–256 | 33,251 | 0.180 | 1.66 |
| 256–512 | 11,464 | 2.898 | 9.23 |
| **합** | **141,027** | | **11.3 시간 (단일 GPU)** |

유휴 GPU 5장으로 나누면 약 2.3 시간이다.  MAIN 전체 추출이 가능하다.
진단전용(JOT 16,814 + 간접주입 3,927)을 더해도 여유가 있다.

## 10. 판정

**PHASE A 통과.**  불변식은 전부 성립하고, float32 추출이 증거를 오염시키지 않음을
float64 기준으로 확인했다.  §49 에 따라 여기서 멈추고 보고한다.

다만 §5 에서 확인된 FP 편중(77.6% 가 한 출처)은 PHASE C/D 설계에 직접 영향을 준다.
pooled 지표만으로는 결론을 낼 수 없다.

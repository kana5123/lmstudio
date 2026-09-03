# PHASE 1 보고: 여러 lightweight guard classifier 를 검증하는 단일 공유 verifier

실행일 2026-09-02.  범위는 지시받은 1~7 단계까지이며 그 이상은 진행하지 않았다.
기존 `direction_correctness` / `direction_debug` / `direction_repro` / DecompX /
`failure_structure` 산출물은 하나도 건드리지 않았다.

---

## 1단계. 모델 레지스트리

이름·아키텍처·라벨 대응을 추측하지 않고 `AutoConfig`, 모델 카드 원문, 그리고
탐침 문장 실측으로 확인했다.

| 모델 | 백본 | 층 | 은닉 | 출력 | 양성 |
|---|---|---:|---:|---|---|
| meta-llama/Llama-Prompt-Guard-2-86M | mDeBERTa-v3-base | 12 | 768 | LABEL_0 / LABEL_1 | 1 (실측 확인) |
| meta-llama/Prompt-Guard-86M | mDeBERTa-v3-base | 12 | 768 | BENIGN / INJECTION / JAILBREAK | 아래 규칙 |
| leolee99/PIGuard (ACL 2025) | deberta-v3-base | 12 | 768 | benign / injection | 1 |
| protectai/deberta-v3-base-prompt-injection-v2 | deberta-v3-base | 12 | 768 | SAFE / INJECTION | 1 |
| deepset/deberta-v3-base-injection | deberta-v3-base | 12 | 768 | LEGIT / INJECTION | 1 |
| fmops/distilbert-prompt-injection | distilbert-base-uncased | **6** | 768 | LABEL_0 / LABEL_1 | 1 (실측 확인) |

확인 과정에서 나온 함정 둘.

**PIGuard 는 은닉표현을 반환하지 않는다.** 커스텀 클래스의 `forward` 가
`output_hidden_states=False` 로 하드코딩돼 있다.  내부에 표준 `DebertaV2Model`
서브모듈이 그대로 있어 직접 호출해 얻었고(모델 파일 수정 없음), 우회 경로의 로짓이
원래 `forward` 와 같은지 배치마다 `assert` 로 확인한다.  실측 최대 차이 9.54e-07.

**PromptGuard v1 의 점수 규칙은 입력 종류에 따라 다르다.** 정상문
"What is the capital of France?" 에 INJECTION 을 1.00 으로 준다.  모델 카드가
이유를 명시한다 — INJECTION 은 제3자 콘텐츠 필터용이고 사용자 프롬프트에는
JAILBREAK 만 쓰라고 되어 있다.  공식 코드도 `get_jailbreak_score = p[2]`,
`get_indirect_injection_score = p[1] + p[2]` 로 나뉜다.  그대로 따랐다.
3-way argmax 를 썼다면 PGv1 이 거의 전부 오탐인 것처럼 보였을 것이다.

## 2단계. 태스크 호환성

60 쌍 판정 — yes 33, partial 18, no 9.  `no` 9 쌍은 전부 **학습셋 누수**다:
PIGuard 는 `piguard:*` 6 종이 자기 학습 데이터, deepset/fmops 는
`prompt-injections`, protectai 는 `jailbreak-classification` 이 자기 학습셋이다.

## 3단계. 추론

호환 51 쌍, **518,064 샘플**.  6개 모델 전부 동결.  층별 CLS 은닉표현(fp16),
로짓, 공격확률, 혼동셀을 저장했다(8.9 GB).

**외부 기준 대조**: PG2 의 BIPIA TPR 이 1/558 로, 이전에 독립적으로 검증한
0.002 와 일치한다.

## 4단계. 혼동 셀 큐브

| 모델 | TP | FP | TN | FN | 오답률 | 오답 중 FN 비율 |
|---|---:|---:|---:|---:|---:|---:|
| deepset | 34,946 | 47,288 | 8,540 | **35** | 0.521 | 0.001 |
| fmops | 34,926 | 48,978 | 6,850 | **55** | 0.540 | 0.001 |
| pgv1 | 17,894 | 14,122 | 42,049 | 17,290 | 0.344 | 0.550 |
| piguard | 17,383 | 7,136 | 29,848 | 9,058 | 0.255 | 0.559 |
| protectai | 17,133 | 5,950 | 49,704 | 17,524 | 0.260 | 0.747 |
| pg2 | 18,895 | 4,243 | 51,928 | 16,289 | 0.225 | 0.793 |

**여섯 모델이 서로 다른 방향으로 실패한다.**  deepset/fmops 는 놓치는 경우가
1000 건에 1 건꼴이고 거의 전부 과탐이다.  pg2/protectai 는 반대로 미탐이 우세하다.
하나의 공유 검증기가 두 방향을 동시에 다뤄야 한다.

오답이 50 개 미만이라 학습에 쓰기 어려운 쌍 2 개(pg2/pgv1 × jailbreak-classification)를
기록해 둔다.

## 5단계. 증거 인터페이스 감사

은닉차원은 6개 모두 768 이라 크기는 맞는다.  층 수는 12 와 6 으로 다르므로
깊이 t = l/L 위에서 7 눈금으로 선형보간해 정렬했다(6층 모델은 항등, 실측 차이 0.0).

**원시 768 차원 기저는 백본 계열을 넘지 못한다.**  깊이별 평균 은닉벡터의 코사인:

| 쌍 | d0 | d2 | d5 | d6(최종) |
|---|---:|---:|---:|---:|
| deepset ↔ piguard | 0.9996 | 1.0000 | 0.9999 | 0.160 |
| deepset ↔ protectai | 0.9989 | 1.0000 | 0.9997 | −0.036 |
| pg2 ↔ pgv1 | 0.9996 | 0.9358 | 0.647 | 0.403 |
| deepset ↔ pg2 | −0.052 | 0.007 | −0.017 | 0.005 |
| fmops ↔ 나머지 전부 | ≈0.02 | ≈0.02 | ≈0.05 | ≈0.03 |

정렬은 **같은 사전학습 초기값을 공유할 때만** 나타난다(deberta-v3-base 계열 3 개,
mDeBERTa 계열 2 개, distilbert 1 개).  계열을 넘으면 코사인이 0 근처다.
최종층에서는 같은 계열 안에서도 정렬이 무너진다 — 미세조정이 마지막 층을 가장 크게 벌린다.

기저 없는 기하 요약(노름·이동량·코사인)은 정의는 공유되지만 스케일이 크게 다르다
(최종층 노름 17.0 ~ 45.1).

## 6·7단계. 모델별 상한 vs 공유 검증기

과제는 base 예측이 맞았는지(TP/TN) 틀렸는지(FP/FN) 맞히기.
분할은 `duplicate_group_id` 단위로 **한 번만** 만들어 6개 모델에 동일 적용했다
(같은 원문을 모델 A 학습·모델 B 평가에 쓰면 누수).  train 362,207 / test 155,857.
지표는 **모델별 AUROC 의 평균** — 6개를 한 통에 넣고 재면 "이 모델은 원래 자주
틀린다" 는 모델 정체성만으로 점수가 오른다.  AUROC 는 절대 뒤집지 않았다.
공유 설정에서는 표준화도 공유한다(모델별 정규화 = 모델별 파라미터라 금지).

| 특징 | 분류기 | 모델별 상한 | **공유(MAIN)** | 차이 |
|---|---|---:|---:|---:|
| conf (확신도, 대조군) | linear | 0.735 | 0.706 | 0.029 |
| conf (대조군) | mlp | 0.749 | 0.717 | 0.032 |
| tfidf (어휘, 대조군) | linear | 0.888 | 0.730 | 0.158 |
| geom (기저 없음) | linear | 0.804 | 0.722 | 0.082 |
| geom (기저 없음) | mlp | 0.888 | 0.824 | 0.064 |
| raw_last (768차원) | linear | 0.942 | 0.879 | 0.063 |
| **raw_last** | **mlp** | **0.971** | **0.961** | **0.010** |

시드 3 회 변동은 표준편차 0.006 이하다.

**base 예측을 고정한 조건부 지표** (지름길 차단; pred1 이 원래 연구 질문):

| 특징·분류기 | pred1 상한 | pred1 공유 | pred0 상한 | pred0 공유 |
|---|---:|---:|---:|---:|
| conf (대조군) linear | 0.703 | 0.703 | 0.797 | 0.797 |
| tfidf (대조군) linear | 0.852 | 0.817 | 0.743 | 0.709 |
| geom mlp | 0.876 | 0.817 | 0.860 | 0.811 |
| **raw_last mlp** | **0.953** | **0.944** | **0.948** | **0.945** |

**라벨 누수 진단**: raw_last mlp 의 점수가 정답 클래스를 맞히는 정도는
AUROC 0.487~0.493 으로 0.5 에서 0.013 이내다.  correctness 를 맞히는 것이지
정답 라벨을 맞히는 것이 아니다.

---

## 무엇이 확인됐고 무엇이 확인되지 않았는가

**확인된 것.**  본 적 있는 6개 모델에 대해, **파라미터 한 벌짜리 공유 검증기가
모델별 전용 검증기의 상한을 거의 그대로 따라간다** (0.961 vs 0.971, 차이 0.010).
원래 연구 질문인 "base 가 UNSAFE 라 했을 때 그게 TP 인가" 에서도 0.944 vs 0.953 이다.
이 값은 확신도 대조군(0.703)과 어휘 대조군(0.817)을 뚜렷이 넘는다.  즉
base 출력의 재탕도 아니고 어휘 신호의 재탕도 아니다.

**확인되지 않은 것 — 이게 더 중요하다.**  은닉벡터만 보고 어느 모델이
만들었는지 **정확도 1.0000 으로 맞힐 수 있다**(기하 특징으로도 0.9932, 우연 0.167).
따라서 공유 MLP 는 파라미터가 한 벌이어도 내부에서 모델별로 갈라 처리하는 것을
학습할 수 있다.  "공유 파라미터" 제약을 문자 그대로는 만족하지만, 본 적 있는
모델에서 높은 점수가 나온 것만으로는 **모델에 무관한 검증** 이 됐다고 말할 수 없다.

그리고 5단계가 그 우려를 뒷받침한다: 원시 768 차원은 **같은 사전학습 초기값을
공유할 때만** 좌표축 의미가 통한다.  계열을 넘으면 코사인이 0 이다.  처음 보는
백본에 그대로 적용될 근거가 없다.

계열을 넘어도 정의되는 기저 없는 기하 특징은 공유 설정에서 0.824 / pred1 0.817 로,
어휘 대조군(0.730 / 0.817)과 **pred1 에서 동률**이다.  일반화 가능한 인터페이스 쪽은
아직 어휘를 넘지 못했다.

판정은 Leave-One-Model-Out 으로만 가능하고, 그것은 PHASE 2 로 보류된 상태다.

## 산출물

```
results/shared_verifier/  model_registry.csv  task_compatibility.csv
                          confusion_cube.csv  probe_results.csv
                          evidence_interface.csv  evidence_basis_alignment.csv
                          evidence_geom_scale.csv
artifacts/shared_verifier/hidden/  51 개 (8.9 GB)
plots/shared_verifier/    phase1_shared_vs_permodel.png
src/shared_verifier/      registry.py infer.py cube.py evidence.py
                          features.py probes.py plots.py
```

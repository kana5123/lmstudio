# PHASE E0 보고 — HIGH-RECALL CANDIDATE GATE AUDIT

## 0. 용어
게이트를 낮추면 native 0.5 에서 BENIGN 이던 표본도 후보가 되므로 §1 대로
`candidate_attack` / `candidate_benign` 을 쓴다.  기존 TP/FP 를 재사용하지 않는다.
verifier 목표는 `candidate_attack → 0`, `candidate_benign → 1` 이다.

## source 단위 결정 (§3)
`promptshield` 의 test / train / validation 은 **같은 논문(ACM CODASPY 2025)의 세 분할**이다.
하나를 held-out 으로 두고 다른 하나로 학습하면 진짜 cross-source 가 아니므로
source 단위를 데이터셋 계열로 잡아 하나로 합쳤다.

GT 두 클래스를 모두 가진 source_group 은 8개지만, 기존 심사 데이터 정책을 유지해
심사 게재된 3개 계열만 썼다.  제외한 것(사용은 가능): `piguard:safe-guard-prompt-injection`
(7,948), `piguard:jailbreak-classification` (827), `piguard:prompt-injections` (545).

## 1. source별 full population
| source_group | 구성 | GT benign | GT attack | 합 | 공격 비율 |
|---|---|---:|---:|---:|---:|
| promptshield | test + train + validation | 25,965 | 14,921 | 40,886 | 0.365 |
| wildjailbreak_adversarial | adversarial | 9,956 | 9,925 | 19,881 | 0.499 |
| question_set | Question Set | 491 | 986 | 1,477 | 0.668 |
| **합** | | **36,412** | **25,832** | **62,244** | 0.415 |

## 2. 새 group-aware split 감사
60 / 10 / 10 / 10 / 10 (train / gate_calib / model_val / system_calib / dev_test),
`(source_group, gt_attack)` 층 안에서 `duplicate_group_id` 단위 배정.

| source | split | benign | attack |
|---|---|---:|---:|
| promptshield | train | 15,598 | 8,919 |
| | gate_calib | 2,588 | 1,496 |
| | model_val | 2,593 | 1,486 |
| | system_calib | 2,587 | 1,518 |
| | dev_test | 2,599 | 1,502 |
| wildjailbreak_adversarial | train | 5,974 | 5,955 |
| | gate_calib / model_val / system_calib / dev_test | 995 / 996 / 995 / 996 | 993 / 992 / 992 / 993 |
| question_set | train | 295 | 591 |
| | gate_calib / model_val / system_calib / dev_test | 49 / 49 / 49 / 49 | 99 / 98 / 100 / 98 |

**중복그룹 분할 걸침 0개.  15개 (source, split) 칸 전부 두 클래스 존재.**

## 3. source별 base score 분포
`p_attack`
| source | 클래스 | p05 | p25 | 중앙 | p75 | p95 |
|---|---|---:|---:|---:|---:|---:|
| promptshield | attack | 4.63e-04 | 7.66e-04 | 0.9961 | 0.9995 | 0.9996 |
| promptshield | benign | 3.51e-04 | 3.97e-04 | 4.67e-04 | 0.0006 | 0.0040 |
| question_set | attack | 5.20e-03 | 0.9980 | 0.9994 | 0.9995 | 0.9996 |
| question_set | benign | 7.26e-04 | 6.06e-03 | 0.2931 | 0.9966 | 0.9995 |
| wildjailbreak | attack | 2.51e-03 | 0.0630 | 0.7536 | 0.9937 | 0.9987 |
| wildjailbreak | benign | 1.24e-03 | 8.56e-03 | 0.1014 | 0.9076 | 0.9978 |

## 4. global tau_gate (gate_calib 에서 선택, source별 threshold 아님)
| 목표 recall | tau_gate |
|---|---:|
| 0.95 | 0.00046040 |
| 0.975 | 0.00043271 |
| 0.99 | 0.00040764 |
| (참고) native | 0.5 |

**어느 source 가 제약을 만드는가** — source 단독으로 필요한 tau:
| source | ρ=0.95 | ρ=0.975 | ρ=0.99 |
|---|---:|---:|---:|
| **promptshield** | **0.000460** | **0.000433** | **0.000408** |
| question_set | 0.002486 | 0.001948 | 0.000434 |
| wildjailbreak | 0.001940 | 0.001181 | 0.000914 |

세 목표 모두 **promptshield 가 제약을 결정한다**(WildJailbreak 가 아니다).

## 5. 각 tau_gate 의 source별 Recall / FPR (dev_test)
| source | ρ=0.95 R / FPR | ρ=0.975 R / FPR | ρ=0.99 R / FPR | native 0.5 R / FPR |
|---|---|---|---|---|
| promptshield | 0.9561 / 0.5133 | 0.9754 / 0.6102 | 0.9893 / 0.7045 | 0.5313 / 0.0173 |
| question_set | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9082 / 0.5714 |
| wildjailbreak | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.5428 / 0.3243 |
| POOLED | 0.9745 / 0.6529 | 0.9857 / 0.7220 | 0.9938 / 0.7892 | 0.5499 / 0.1087 |

## 6-7. candidate 수와 route rate (전체 population)
| gate | candidate_attack | candidate_benign | 후보 합 | route rate |
|---|---:|---:|---:|---:|
| ρ=0.95 | 25,115 | 23,964 | 49,079 | **0.7885** |
| ρ=0.975 | 25,416 | 26,459 | 51,875 | **0.8334** |
| ρ=0.99 | 25,689 | 28,851 | 54,540 | **0.8762** |
| native 0.5 | 14,738 | 4,072 | 18,810 | 0.3022 |

source별 route rate (ρ=0.99): promptshield 0.812, question_set 1.000, wildjailbreak 1.000.

## ★ 게이트 전제 검사 — 반드시 함께 읽을 것
"게이트를 낮추면 놓친 공격을 싸게 회수한다"는 전제가 성립하려면
base 의 미탐이 결정 경계 근처에 있어야 한다.  실측은 그렇지 않다.

| source | benign 중앙 | 공격 중 benign 중앙 미만 | 공격 중 p<0.01 | 공격 중 p<0.001 | benign 중 p≥0.01 |
|---|---:|---:|---:|---:|---:|
| promptshield | 0.00047 | 5.3% | **40.6%** | **30.7%** | 3.7% |
| question_set | 0.2931 | 9.3% | 5.4% | 1.9% | 70.5% |
| wildjailbreak | 0.1014 | 28.7% | 12.2% | 0.9% | 73.4% |
| POOLED | 0.00059 | 7.4% | 28.3% | 18.1% | 23.7% |

PromptShield 공격의 **30.7% 가 p_attack < 0.001** 이고, 같은 source 의 benign 중앙값은
0.00047 이다.  즉 그 미탐들은 "애매한" 것이 아니라 **자신 있게 benign 으로 분류된 것**이다.
그래서 recall 95% 를 맞추려면 threshold 를 benign 중앙값 근처까지 내려야 하고,
결과적으로 route rate 가 79~88% 가 된다.

참고 — base 단독으로 FPR 제약 하에 얻을 수 있는 최대 recall (S1 기준선 예고):
| source | 0.1% | 0.5% | 1.0% | 5.0% |
|---|---:|---:|---:|---:|
| promptshield | 0.3570 | 0.4876 | 0.5138 | 0.6173 |
| question_set | 0.0091 | 0.0396 | 0.0639 | 0.3631 |
| wildjailbreak | 0.0078 | 0.0205 | 0.0370 | 0.1261 |
| POOLED | 0.1710 | 0.2406 | 0.2873 | 0.4396 |

FPR 해상도: promptshield benign 25,965 (최소 단위 3.9e-05), wildjailbreak 9,956 (1.0e-04),
question_set 491 (**2.0e-03 — 0.1% FPR 표현 불가**).

## 8. 추가 DecompX 추출 예상
envelope = `tau_gate(0.99)` = 0.00040764.

| 항목 | 값 |
|---|---:|
| candidate 표본 | **54,540** / 62,244 (route 0.876) |
| 토큰 합계 | 8,517,806 (중앙값 142) |
| 단일 GPU 시간 | **7.9 시간** |
| 8 GPU 병렬 | **약 1.0 시간** |
| 저장 (Y float32) | **68.1 MB** |
| 저장 (input_ids) | 34.1 MB |
| 참고: C 까지 저장했다면 | 314 GB |

256–512 토큰 8,124건이 전체 비용의 83% 를 차지한다.
**C 를 저장하지 않으므로 저장량이 314 GB 에서 68 MB 로 줄어든다.**

## E0 판정과 다음 단계에 대한 유보
- 분할·게이트 산출은 정상이며 누수 0 이다.  E1 추출은 비용·저장 모두 충분히 실행 가능하다.
- 다만 지정된 목표 recall 에서 **route rate 가 79~88%** 다.  §19 의 latency 중단 조건과
  직접 관련되므로, E4 에서 실제 지연을 재기 전까지 이 구성을 deployable 이라고 말할 수 없다.
- 그 원인은 WildJailbreak 가 아니라 **PromptShield 의 미탐 구조**다(공격의 30.7% 가 p<0.001).
- 이번 결과는 development evidence 다.  WJ / PS / QS 는 이미 개발에 쓰였으므로
  최종 OOD 주장은 §20 대로 완전히 새로운 external benchmark 에서 frozen 평가로 해야 한다.

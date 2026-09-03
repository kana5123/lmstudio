# Base PromptGuard2 source별 OOD 감사

목적: PHASE C1 의 LOSO verifier 실패가 (1) base detector 자체의 OOD 실패와 함께 일어나는지,
(2) base 는 잘 작동하는데 verifier 만 source 일반화에 실패하는지 가른다.

## 두 AUROC 의 구분

| | BASE DETECTOR AUROC | VERIFIER AUROC |
|---|---|---|
| 모집단 | 해당 source 의 GT benign + GT attack **전부** (TP/FP/TN/FN) | base 가 ATTACK 이라 예측한 것만 (TP+FP) |
| 양성 | GT attack | FP |
| 점수 | `z_attack − z_benign` | verifier `P(FP)` |

PHASE C1 의 M0/A0/A3 는 **전부 후자**다.  M0 를 base detector AUROC 라고 부르지 않는다.
`p_attack` 은 이진 softmax 이므로 margin 과 랭킹이 동일하다
(spearman 1.0000000000, `sigmoid(margin)` 대 `p_attack` 최대차 1.19e-07).
두 점수를 곱하거나 더해 combined AUROC 를 만들지 않았다.

## 1-2. source별 population 과 혼동셀

| source_group | GT benign | GT attack | n | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| piguard:Question Set | 491 | 986 | 1,477 | 888 | 236 | 255 | 98 |
| promptshield:test | 16,522 | 5,506 | 22,028 | 1,540 | 439 | 16,083 | 3,966 |
| wildjailbreak:adversarial | 9,956 | 9,925 | 19,881 | 5,470 | 3,374 | 6,582 | 4,455 |

(CORE 캐시는 TP+FP 만 담고 있어 base AUROC 계산에 쓰지 않았다.)

## 3-4. Base AUROC/AUPRC 와 native threshold

| source_group | AUROC | AUPRC | Recall | FPR | Precision | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| promptshield:test | **0.8557** | 0.6308 | 0.2797 [0.268,0.292] | 0.0266 [0.0242,0.0291] | 0.7782 | 0.9734 |
| piguard:Question Set | 0.8301 | 0.8964 | 0.9006 [0.880,0.918] | **0.4807** [0.437,0.525] | 0.7900 | 0.5193 |
| wildjailbreak:adversarial | **0.6469** | 0.6322 | 0.5511 [0.541,0.561] | 0.3389 [0.330,0.348] | 0.6185 | 0.6611 |

## 5. Recall @ FPR 제약

제약을 만족하는 threshold 중 **Recall 이 최대**가 되는 지점을 골랐다.
동률 처리: `sklearn.roc_curve` 는 같은 점수의 표본을 하나의 threshold 로 묶으므로,
선택된 threshold 에서 동점 표본은 모두 함께 양성으로 들어간다(부분 포함 없음).

| source | 목표 FPR | threshold | 달성 FPR | 달성 Recall [95% CI] | TP | FP | benign 분모 | attack 분모 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Question Set | 0.1% ★ | 7.7952 | 0.0000 | 0.0091 [0.005,0.017] | 9 | 0 | 491 | 986 |
| Question Set | 0.5% | 7.7790 | 0.0041 | 0.0396 [0.029,0.054] | 39 | 2 | 491 | 986 |
| Question Set | 1.0% | 7.7676 | 0.0082 | 0.0649 [0.051,0.082] | 64 | 4 | 491 | 986 |
| promptshield | 0.1% | 7.0519 | 0.0010 | 0.0171 [0.014,0.021] | 94 | 16 | 16,522 | 5,506 |
| promptshield | 0.5% | 6.7869 | 0.0050 | 0.0312 [0.027,0.036] | 172 | 82 | 16,522 | 5,506 |
| promptshield | 1.0% | 5.5716 | 0.0099 | **0.1787** [0.169,0.189] | 984 | 164 | 16,522 | 5,506 |
| wildjailbreak | 0.1% | 7.2021 | 0.0008 | 0.0078 [0.006,0.010] | 77 | 8 | 9,956 | 9,925 |
| wildjailbreak | 0.5% | 6.9864 | 0.0049 | 0.0205 [0.018,0.023] | 203 | 49 | 9,956 | 9,925 |
| wildjailbreak | 1.0% | 6.7925 | 0.0099 | 0.0370 [0.033,0.041] | 367 | 99 | 9,956 | 9,925 |

★ = 통계 해상도 부족.

## 6. FPR 통계 해상도

| source | benign n | 최소 FPR 단위 (1/n) | 0.1% 에서 허용 FP | 해상도 부족? |
|---|---:|---:|---:|---|
| piguard:Question Set | 491 | 0.002037 | 0 | **예** (0.1% < 최소단위) |
| promptshield:test | 16,522 | 0.000061 | 16 | 아니오 |
| wildjailbreak:adversarial | 9,956 | 0.000100 | 9 | 아니오 |

Question Set 은 benign 이 491개뿐이라 FP 1개가 0.204% 다.  0.1% 를 표현할 수 없다.

## 7. base score (`z_attack − z_benign`) 분포

| source | 클래스 | n | mean | median | p05 | p25 | p75 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Question Set | benign | 491 | 0.154 | −0.880 | −7.227 | −5.101 | **5.689** | 7.592 |
| Question Set | attack | 986 | 5.787 | 7.393 | −5.255 | 6.047 | 7.679 | 7.775 |
| promptshield | benign | 16,522 | −7.025 | −7.611 | −7.947 | −7.788 | −7.307 | −4.072 |
| promptshield | attack | 5,506 | −3.117 | **−6.516** | −7.526 | −7.253 | 2.601 | 6.614 |
| wildjailbreak | benign | 9,956 | −1.253 | −2.182 | −6.694 | −4.752 | 2.284 | 6.104 |
| wildjailbreak | attack | 9,925 | 0.948 | 1.118 | −5.984 | −2.700 | 5.066 | 6.681 |

## 8. Base 와 C1 verifier LOSO 나란히 (합성하지 않음)

| held-out source | Base AUROC (attack vs benign) | Base Recall@1%FPR | M0 FP-vs-TP | A0 FP-vs-TP | A3 FP-vs-TP |
|---|---:|---:|---:|---:|---:|
| promptshield:test | **0.8557** | 0.1787 | 0.6167 | 0.7068 | **0.3096** |
| piguard:Question Set | 0.8301 | 0.0649 | 0.5507 | 0.7037 | 0.5115 |
| wildjailbreak:adversarial | 0.6469 | 0.0370 | 0.5742 | 0.5113 | 0.4810 |

## 9. source별 해석

**promptshield:test — CASE 1 (base 는 일반화, trajectory verifier 만 source-specific)**
base AUROC 가 셋 중 가장 높고(0.856) native FPR 도 2.66% 로 낮다.  그런데 이 source 에서
A3 가 0.310 으로 가장 나쁘다.  base representation 의 분포 이동으로는 설명되지 않는다.
다만 base 의 native recall 은 28.0% 에 그쳐 운영점 자체는 미탐 쪽으로 크게 치우쳐 있다.

**wildjailbreak:adversarial — CASE 2 (base 와 verifier 가 함께 약함)**
base AUROC 가 0.647 로 가장 낮고 native FPR 33.9% 다.  점수 분포에서 benign 중앙값 −2.18,
attack 중앙값 +1.12 로 크게 겹친다.  M0/A0/A3 가 모두 0.48~0.57 로 우연 근처다.
이 source 는 hard negative 를 의도적으로 만든 집합이므로 base 가 어려운 것이 설계상 자연스럽다.

**piguard:Question Set — 중간**
base AUROC 0.830 으로 랭킹은 괜찮지만 native FPR 이 48.1% 로 운영점이 무너져 있다
(benign p75 가 +5.69 로 정상문 상당수가 높은 점수를 받는다).  A0 는 0.704 로 작동하지만
A3 는 0.512 로 우연 수준이다.

**전체 패턴**
base AUROC 와 A3 LOSO 성능이 같은 방향으로 움직이지 않는다.
base 가 가장 좋은 source(promptshield 0.856)에서 A3 가 가장 나쁘다(0.310).
따라서 PHASE C1 의 LOSO 실패를 "base 가 OOD 에서 무너졌기 때문"으로 설명할 수 없다.
CASE 1 이 최소 한 source 에서 명확히 성립하며, CASE 3(verifier 가 base 를 넘는 cross-source
보정 신호를 갖는 경우)은 A3 에서는 관측되지 않았다.  다만 A0 는 base AUROC 가 낮은
promptshield·Question Set 에서 0.70 대를 유지해, 최종 attribution 만은 일부 cross-source
신호를 갖는다는 §C1 관찰과 일치한다.

## 하지 않은 것
final cascade AUROC 를 만들지 않았다.  두 점수를 곱하거나 더하지 않았다.
새 학습, ablation, 추가 DecompX 추출을 수행하지 않았다.

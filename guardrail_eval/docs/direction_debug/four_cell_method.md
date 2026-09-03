# §4~§9 — 2×2 혼동 셀 방법

## 정의 (전부 CORRECT − INCORRECT 방향으로 통일)

`g_i^(l) = h_CLS,i^(l) − h_CLS,i^(l-1)` 의 cell 평균 `mu_TP, mu_FP, mu_TN, mu_FN` 에서:

```
delta_U    = mu_TP − mu_FP                          UNSAFE 가지: 맞음 − 틀림
delta_S    = mu_TN − mu_FN                          SAFE   가지: 맞음 − 틀림
delta_GT   = 0.5[(mu_TP+mu_FN) − (mu_FP+mu_TN)]     정답 클래스 (공격 − 정상)
delta_PRED = 0.5[(mu_TP+mu_FP) − (mu_FN+mu_TN)]     모델 예측 (UNSAFE − SAFE)
delta_CORR = 0.5[(mu_TP+mu_TN) − (mu_FP+mu_FN)]     정오 상호작용 (맞음 − 틀림)
```

cell 표본 수가 크게 다르므로 **pooled mean 을 쓰지 않고 각 cell 평균에 동일 가중**을 준다.

## §7 독립 교차검증 (구현 오류 방지)

각 표본에 `GT ∈ {+1 공격, −1 정상}`, `PRED ∈ {+1 UNSAFE, −1 SAFE}` 를 부여하면
`GT × PRED` 가 정확히 정오가 된다 (TP:+1, TN:+1, FP:−1, FN:−1).

`g_i = b + β_GT·GT + β_PRED·PRED + β_CORR·(GT·PRED) + ε` 를
**cell 당 총가중이 같도록 가중최소제곱**으로 적합했다.

결과: `cos(β_CORR, delta_CORR)` 최소값 **1.000000** — 대비 코딩에 오류 없음.

## 표본 상한

cell 평균만 필요하므로 cell × split_role 당 **3,000건 상한**(seed 0)을 뒀다.
상한 적용 전후 개수는 `results/direction_debug/cell_counts.csv` 에 둘 다 있다.

## TRAIN cell 수 (상한 적용 후)

| dataset | TP | FP | TN | FN | 비고 |
|---|---:|---:|---:|---:|---|
| wildjailbreak | 3000 | 2413 | 3000 | 3000 | 네 cell 모두 충분 |
| promptshield_test | 1260 | 271 | 3000 | 3000 | 충분 |
| questionset | 1066 | 270 | 193 | **71** | ★ TN/FN 작음 + FP 라벨 오염 확인됨 → **불안정** |
| promptshield_train | 3000 | **18** | 3000 | 1704 | ★ FP 18 → 불안정, 독립 데이터셋 아님(같은 데이터셋 다른 split) |
| jailbreaksovertime | 2207 | 1380 | 3000 | **70** | ★ FN 70 → `delta_S` 불안정. 말뭉치 교란 확정 → 대조군 |

**독립 데이터셋으로 세는 것은 wildjailbreak / promptshield_test / questionset 3개뿐.**

## 신뢰도는 통제 전용

`logit_unsafe`, `logit_margin`, `unsafe_probability` 는 저장했으나
**어떤 방향의 정의에도 쓰지 않았다.** 오직 §9 진단(사영과의 Spearman)에만 사용한다.

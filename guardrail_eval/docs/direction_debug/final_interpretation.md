# 최종 해석 — §23 taxonomy 및 §24 질문 10개

## 판정: **TYPE A — CLASS SEMANTIC SIGNAL**

부차적으로 후반 두 층(L10→L12)에 한해 **TYPE B — PREDICTION / CONFIDENCE SIGNAL** 이
겹친다. **TYPE D(cross-dataset correctness signal)는 명확히 기각된다.**

### TYPE A 판정 근거 (§23 조건과 대조)

| §23 TYPE A 조건 | 실측 |
|---|---|
| `delta_U ≈ −delta_S` | WildJailbreak L1→L10 에서 **−0.83 ~ −1.00** (L1→L2 는 −0.996) |
| `delta_U` 가 `delta_GT` 와 강하게 정렬 | WildJailbreak **0.968~0.999**, PromptShield **0.80~0.97** |
| `delta_CORR` 약함 | 크기 **0.00~0.03** (같은 층 `delta_GT` 는 0.02~0.19), cross-dataset 전이 우연 이하 |

`cos(delta_U, delta_S) = −1` 은 산수로 설명된다. UNSAFE 가지에서 TP=정답공격·FP=정답정상이라
`delta_U ≈ (공격−정상)` 이고, SAFE 가지에서 TN=정답정상·FN=정답공격이라
`delta_S ≈ (정상−공격) = −(공격−정상)` 이다. **정오 성분이 사실상 없다.**

---

## §24 질문 10개에 대한 답

### 1. 기존 WJ TP-FP AUROC 0.72 는 class signal 인가 correctness signal 인가?

**class signal 이다.** `cos(delta_U, delta_GT)` 가 L1→L10 에서 **0.968~0.999**.
`delta_U` 는 정답 공격/정상 축과 사실상 같은 방향이다.

### 2. `delta_U` 와 `delta_S` 는 같은 방향인가 반대 방향인가?

**반대 방향이다.** WildJailbreak −0.83~−1.00, PromptShield_test −0.40~−0.95 (L1→L10).
치환 귀무(정오 라벨만 섞음, N=10,000)를 크게 벗어난 **음의** 값이다.
QuestionSet 만 초반에 양수(+0.74)인데, TN 117 / FN 42 로 cell 이 작고
FP 라벨 오염이 확인된 데이터셋이라 신뢰하지 않는다.

### 3. `delta_CORR` 가 실제로 존재하는가?

**사실상 존재하지 않는다.** 세 근거가 모두 같은 방향을 가리킨다.
- 크기가 0 에 가깝다: WildJailbreak L1→L10 에서 `||delta_CORR|| = 0.00~0.03`
- 데이터셋 간 방향이 일치하지 않는다: 세 쌍이 같은 부호로 맞는 층이 하나도 없음
- 화이트닝하면 미미한 양의 정렬(+0.079)조차 음수(−0.047)가 된다

### 4. 어느 layer 에서 가장 강한가?

`delta_CORR` 만 놓고 보면 **후반 L10→L12** 에서 크기가 커진다(1.21 → 2.09).
그러나 그 구간은 5번 답처럼 결정축과 구별되지 않는다.
초·중반에서는 어느 층에서도 유의미하지 않다.

### 5. 그 layer 가 PG2 confidence axis 와 동일한가?

**거의 동일하다.** WildJailbreak L10→L11 / L11→L12 에서
사영과 PG2 logit margin 의 Spearman:
`v_PRED` **0.985 / 0.998**, `v_GT` **0.983 / 0.998**.
게다가 `||delta_PRED||` 가 **14.85 / 31.18** 로 다른 효과를 한 자릿수 압도한다.
**후반 층의 "신호"는 새 내부 정보가 아니라 모델의 결정축이다.**

### 6. JOT 를 제외한 최소 3개 dataset 에서 `delta_CORR` 가 재현되는가?

**아니다.** wildjailbreak / promptshield_test / questionset 세 쌍의 부호 있는 cos 는
층마다 뒤집히고 대부분 귀무 95%ile 안에 있다. 유의한 것은 오히려 **반정렬**
(`PS_test|QS` L1→L2 = −0.932, signed p = 0.012)이다.

### 7. 한 dataset 에서 fit 한 correctness direction 이 다른 dataset 으로 transfer 되는가?

**아니다.** `delta_CORR` 전이 AUROC 가 0.26~0.76 이고 대부분 0.5 근처이며,
0.270 / 0.271 / 0.286 처럼 **우연을 크게 밑도는** 칸이 여럿이다
(= 다른 데이터셋에서 반대로 작동). 부호를 뒤집지 않고 그대로 보고한 값이다.

### 8. LODO 에서도 남는가?

**남지 않는다.** WildJailbreak held-out 0.487~0.565(전 층 우연),
PromptShield held-out 0.274~0.663(대부분 0.5 미만),
QuestionSet 0.458~0.736(불안정 데이터셋).

### 9. shuffle null 의 높은 cosine 은 common mean 때문인가, covariance anisotropy 때문인가?

**공분산 비등방성 / 낮은 유효차원 때문이다. 이전의 "common mean" 설명은 틀렸다.**
평균차에서 공통 평균은 상쇄된다. 실측: 이동벡터의 **유효 랭크가 768 중 1.2~11.4**,
상위 1개 주성분이 분산의 26~91%를 설명한다.
**spearman(유효랭크, 귀무|cos|) = −0.927**, pearson(1/유효랭크, 귀무|cos|) = +0.907.

### 10. 현재 결과로 DecompX token-level 분석을 correctness mechanism 으로 진행할 근거가 있는가?

**없다.** §20 게이트 4개 조건 중 **하나도 통과하지 못했다.**

| 게이트 조건 | 결과 |
|---|---|
| 1. `delta_CORR` 가 held-out 에서 유의 | **실패** — 크기 ~0, 유의하지 않음 |
| 2. `delta_U` 와 `delta_S` 가 같은 orientation | **실패** — cos ≈ −1 (정반대) |
| 3. 3개 이상 dataset 에서 cross-dataset 일치 | **실패** — 부호가 뒤집힘 |
| 4. LODO 에서 meaningful transfer | **실패** — 우연 또는 그 이하 |

따라서 DecompX 의 `D_k^(l)` / `a_k^(l)` 토큰 히트맵을 **correctness mechanism 으로
해석하지 않는다.** 보존된 conservation identity 검증(`sum_k D_k = g`, 상대오차 1e-6~1e-4,
코사인 ≥ 0.99999964)은 수학적 성질이므로 그대로 유효하다.

---

## 한계 — 이 결론이 뒤집힐 수 있는 지점

1. **QuestionSet 은 신뢰할 수 없다.** TN 117 / FN 42, FP 라벨 오염(탈옥 표지 5.8배 농축).
   세 데이터셋 중 하나가 이 상태라 "3개 독립 데이터셋" 이라는 요건이 실질적으로는 2.5개다.
2. **PromptShield 는 어휘만으로 TP/FP 가 AUROC 1.000 분리된다.** 코퍼스 지문이 강해
   그 데이터셋에서의 방향은 문체를 반영할 수 있다.
3. **PG2 는 FP 를 거의 만들지 않는다.** `BENIGN` FPR 0.73% 대 hard negative 31.6%.
   그래서 FP cell 이 큰 데이터셋 자체가 희소하고, 이는 표본 편향을 낳는다.
4. **`delta_S` 는 FN cell 에 의존하는데** JOT 는 FN 70, QuestionSet 은 42 로 매우 작다.
   WildJailbreak(FN 3000)과 PromptShield_test(FN 3000)만 이 축에서 신뢰할 만하다.
5. **더 강한 반증 설계**: 같은 데이터셋 안에서 GT 를 고정한 채 PG2 정오만 달라지는
   짝(예: 거의 동일한 두 공격문 중 하나만 탐지)을 모으면 `delta_CORR` 를
   class semantics 로부터 훨씬 깨끗하게 분리할 수 있다. 현재는 그런 짝을 만들지 않았다.

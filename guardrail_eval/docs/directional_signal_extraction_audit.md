# 방향성 신호 추출 감사 (Directional Signal Extraction Audit)

목적: **"DecompX 기반 층별 방향성 신호가 정탐(TP)과 오탐(FP) 사이에 실제로 존재하는가"**
를 검증하기 전에, 우리가 필요한 텐서가 무엇이고 기존 산출물이 그것을 갖고 있는지 확정한다.

---

## 0. 이전 분석과 이번 분석은 다른 신호다 (정정)

앞서 보고한 **"상위 UNSAFE-margin 토큰의 86.2% / 86.4% 가 구두점"** 결과는
**이번 방법론의 신호가 아니다.**

| | 이전 분석 | 이번 분석 |
|---|---|---|
| 대상 텐서 | `out.classifier` — 분류기 로짓에 대한 토큰 귀속 | `out.cls_encoder` — **층 최종 인코더 분해**의 CLS 행 |
| 코드 위치 | `modeling_deberta_v2_decompx.py:315` (`classifier.weight` 곱) | 같은 파일 `:298` (`attrib[:, 0]`, post-LN2 시점) |
| 수식 | `A[k,c]`, 그리고 `margin_k = A[k,UNSAFE] - A[k,BENIGN]` | `C_k^(l)`, 그리고 `D_k^(l) = C_k^(l) - C_k^(l-1)` |
| 무엇을 묻나 | "어떤 토큰이 UNSAFE-vs-benign **로짓 여유**를 밀었나" | "토큰 k 에 귀속된 CLS 기여가 **인코더 블록 하나를 지나며 어떻게 움직였나**" |
| 방향 사용 | 없음 (스칼라 여유) | `v_U^(l)` (학습셋 TP-FP 평균차 방향)에 사영 |

**따라서 구두점 결론을 이번 신호에 그대로 옮기지 않는다.** 17절에서 이번 신호에 대해
독립적으로 다시 측정한다.

---

## 1. 우리가 필요한 텐서의 정의

DecompX 원본(`third_party/DecompX/src/modeling_bert.py`)에서 **층 최종 인코더 분해**는
attention 만의 출력이 아니라

```
attention → residual 1 → LN1 → FFN → residual 2 → LN2
```

를 전부 통과한 `post_ln2_layer` 다 (`modeling_bert.py:823-829`, 그리고 `:836` 에서
`encoder=output_builder(post_ln2_layer, "both")` 로 내보낸다).

우리의 `C_k^(l)` 는 이에 대응하는 DeBERTaV2 층 최종 분해의 **CLS 대상 위치 행**이다.

```
C_i^(l)  shape = [num_source_tokens, hidden_dim]
C_i,k^(l) = 표본 i 의 층 l 최종 CLS 표현에서 입력토큰 k 에게 귀속된 은닉차원 벡터
```

**사용 금지 목록** (지시문 1절): 주의집중 확률, attention×value only, 분류기 기여,
UNSAFE 로짓 기여, 분류기 여유 기여, FFN 만의 기여.

---

## 2. 우리 포팅이 올바른 텐서를 계산하는가 — YES

`src/pg2_decompx/modeling_deberta_v2_decompx.py` 의 `_layer()` 는 원본과 같은 순서로

```python
a = ln_decomposer(res_w, pre_ln1, ...)          # LN1   (:253)
a = a + self._ffn_decompose(...)                # FFN + 잔차2 (:260)
a = ln_decomposer(a, pre_ln2, ...)              # LN2   (:262)
return layer_out, a                             # a = post_ln2_layer
```

를 수행하고, `forward()` 가 층마다

```python
per_layer.append(attrib[:, 0].clone())          # (:298)  CLS 행만 보관
out.cls_encoder = torch.stack(per_layer, dim=1) # (:309)  (B, L, N, H)
```

로 쌓는다. 즉 **`cls_encoder[:, l-1]` = 층 l 의 `post_ln2_layer` 의 CLS 행 = `C_k^(l)`** 이다.

---

## 3. 기존 6,142표본 / 32샤드 산출물 감사 — 이번 실험에 **쓸 수 없다**

`artifacts/features/decompx_{split}_{i}of8.pt` 가 실제로 담고 있는 것:

| 키 | 모양 | 정체 |
|---|---|---|
| `delta_c` | `(n, 512, 768)` fp16 | **`C^(12) - C^(1)` 하나뿐** — 12개 층이 하나로 접혀 있다 |
| `directional` | `(n, 512)` | `dot(delta_c_k, v)` — 그런데 `v` 는 `delta_h = h_L - h_1` 로 만든 **층 무관 단일 방향** |
| `margin` | `(n, 512)` | `A[k,UNSAFE] - A[k,BENIGN]` — **분류기 귀속**(이번 실험 금지 목록) |
| `mask`, `input_ids`, `gt`, `seq_len` | | 재사용 가능 |
| `recon_rel_err` | `(n,)` | `sum_k delta_c_k` vs `h^(12)-h^(1)` **한 쌍만**, 그것도 최대절대비 기준 |

**판정: 층별 `C_k^(l)` 이 없다. 재추출 필요.**

기존에 보고한 `복원 상대오차 max 6.68e-04 / mean 5.86e-05` 는
**최종 층 한 쌍(`h^(12)-h^(1)`)에 대한 최대절대값 비율**이며,
**층별 L2 상대오차가 아니다.** 아래 4절이 그것을 새로 측정한 값이다.

재사용 가능한 것: `artifacts/features/hidden_{split}.pt` 에 이미
**모든 층의 CLS 은닉표현** `h` `(n, 13, 768)` 가 있다 (0=임베딩, 1..12=인코더 층).
따라서 `g^(l)` 와 `v_U^(l)` 는 **재추출 없이 즉시** 계산할 수 있다.

---

## 4. 전제 검증 결과 (실측, `src/directional/verify_layer_semantics.py`)

### (1) 분류 대상 위치 = 색인 0 = `[CLS]`

```
토큰열: ['[CLS]', '▁hello', 'world', '[SEP]']
cls_token='[CLS]' id=1 | sep='[SEP]' id=2 | pad_id=0
input_ids[0]==1? True
ContextPooler.forward:  context_token = hidden_states[:, 0]
                        pooled_output = self.dense(context_token)
```

관습이 아니라 토크나이저·풀러 소스로 확인했다.

### (2) `hidden_states` 색인

```
len(hidden_states)=13  (층수 12 + 1)
hidden_states[0]  vs 임베딩 출력       max diff = 0.000e+00
hidden_states[12] vs last_hidden_state max diff = 0.000e+00
```

### (3) 층별 복원 `sum_k C_k^(l)` vs `h_CLS^(l)` — **12개 층 전부 성립**

| 층 | float32 상대 L2 오차 | float64 상대 L2 오차 |
|---:|---:|---:|
| 1 | 4.99e-07 | 8.89e-14 |
| 2 | 4.49e-07 | 1.16e-13 |
| 3 | 4.22e-07 | 3.14e-13 |
| 4 | 8.71e-07 | 4.60e-13 |
| 5 | 8.75e-07 | 3.47e-13 |
| 6 | 8.75e-07 | 3.96e-13 |
| 7 | 1.23e-06 | 4.93e-13 |
| 8 | 9.77e-07 | 5.31e-13 |
| 9 | 2.69e-06 | 1.89e-12 |
| 10 | 7.68e-06 | 7.93e-12 |
| 11 | 3.14e-04 | 2.81e-10 |
| 12 | 1.84e-04 | 1.65e-10 |

float64 로 올리면 오차가 같이 줄어든다 → **구조적 근사가 아니라 부동소수점 반올림.**

### (4) 패딩·특수토큰 규약

```
패딩 위치 기여 노름 최대 = 0.000e+00
전체 위치 합 == 유효 토큰만의 합  (모든 층에서 자릿수까지 동일)
```

DecompX 의 `bias_decomposer` 는 편향을 **원천토큰들에게 정규화 가중치로 나눠** 넣으므로
별도 편향 항이 남지 않는다. 따라서 등식은 `sum_k C_k^(l) = h_CLS^(l)` 그대로다.
`[CLS]`/`[SEP]` 는 **합산에 포함**해야 등식이 성립하며(그들도 원천 토큰이다),
사람이 보는 표에서만 따로 구분한다.

---

## 5. 재추출 계획 (58 GB 를 피하는 설계)

모든 층의 `C_k^(l)` 를 그대로 저장하면 `6142 × 12 × 512 × 768 × 2 B ≈ 58 GB` 다.
불필요하다. `v_U^(l)` 는 **일반 은닉표현만으로** 먼저 확정할 수 있으므로
(3절), 추출 시점에 이미 `v_U^(l)` 를 알고 있다. 따라서 추출 중에

```
a_i,k^(l) = dot( v_U^(l), D_i,k^(l) )        # (11, N) — 표본당 22 KB
```

를 **바로 계산해 저장**하고, 검증에 필요한 스칼라

```
층별 복원오차,  || sum_k D_k^(l) - g^(l) ||,  cos,  sum_k a_k^(l) vs dot(v,g)
```

도 함께 남긴다. 원본 `C` 는 **대표 표본 소수(히트맵용)** 에만 저장한다.
→ 총 저장량 수백 MB 수준.

**중요**: `v_U^(l)` 는 `ver_train` 의 TP/FP 에서만 적합하며, 추출 대상에 held-out 이
포함되어도 방향 자체는 held-out 을 보지 않는다.

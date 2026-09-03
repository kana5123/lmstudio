# DecompX → DeBERTaV2 포팅 분석서 (Llama-Prompt-Guard-2-86M)

작성 기준: 아래 세 파일을 **직접 읽고 확인**한 내용만 기록한다. 7명의 리더 보고서 중
소스와 어긋나는 주장은 소스를 따르고, 어긋난 지점은 본문에 명시했다.

| 대상 | 경로 | 확인 |
|---|---|---|
| DecompX 설정/출력 자료구조 | `third_party/DecompX/src/decompx_utils.py` (50줄) | 전문 읽음 |
| DecompX BERT 구현 | `third_party/DecompX/src/modeling_bert.py` (2452줄) | 분해 관련 전 구간 읽음 |
| DeBERTaV2 원본 | `/home/kana5123/ETRI/.venv/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py` (transformers **4.51.3**) | 전 구간 읽음 + **실제 실행 검증** |
| 대상 체크포인트 config | `~/.cache/huggingface/hub/models--meta-llama--Llama-Prompt-Guard-2-86M/snapshots/a8de.../config.json` | 전문 읽음 |

> 보고서 정정 3건 (소스가 이김):
> 1. 설정 필드 이름은 `bias_decomp_type` 이고 `biases_decomp_type` 는 **없다**.
> 2. 풀러 활성 근사 필드 이름은 `tanh_approx_type` 이고 `tanh_approx_pooler` 라는 심볼은 **없다**.
> 3. `decompx_utils.py:25` 의 `tanh_approx_type` 기본값은 **`"ZO"`** 다 (한 리더가 `"LA"` 라고 적었으나 오류).
>    단, `BertModel.ffn_decomposer` 의 파이썬 기본 인자만 `tanh_approx_type="LA"` 이다 (`modeling_bert.py:1347`).
> 4. `output_builder` 의 네 번째 모드 문자열은 `"distance_based"` 이며 `"dist"` 는 없다 (`modeling_bert.py:174`).

---

## 1. DecompX가 무엇을 계산하는가

DecompX는 **입력 토큰 하나하나가 어떤 벡터로 최종 표현/로짓에 기여했는지**를 추적한다.
어텐션 가중치 지도(attention map)처럼 "얼마나 봤나"가 아니라, **은닉 상태를 입력 토큰별
기여 벡터들의 합으로 완전히 쪼갠다**는 점이 핵심이다.

핵심 객체는 **기여 벡터 묶음(attribution vectors)** 이며, 인코더 내부에서는 항상 4차원이다.

```
attribution_vectors : (B, N_i, N_j, D)
  B   = 묶음 크기 (batch)
  N_i = 목적지 토큰 i (destination) — "누구의 은닉 상태를 설명하는가"
  N_j = 출처 토큰 j (source)        — "그 은닉 상태의 몇 %가 j 로부터 왔는가"
  D   = 은닉 차원 (hidden_size, BERT-base=768)
```

축 의미는 원저자가 직접 못박아 놨다:

```python
# third_party/DecompX/src/modeling_bert.py:538-542
def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
    # Decomposes the input bias based on similarity to the attribution vectors
    # Args:
    #   bias: a bias vector (all_head_size)
    #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size)
    #                        :: (batch, seq_length, seq_length, all_head_size)
```

**불변식(invariant)** — 이 문서 전체를 관통하는 단 하나의 성질:

```
sum over j  of  attribution_vectors[b, i, j, :]   ==   hidden_states[b, i, :]
(출처 축 dim=-2 로 더하면 진짜 은닉 상태가 그대로 복원된다)
```

층을 지나며 축이 줄어드는 지점:

| 단계 | 모양 | 축소가 일어나는 코드 |
|---|---|---|
| 인코더 각 층 출력 | `(B, N, N, D)` | — |
| 풀러(pooler) 입력 | `(B, N, D)` | `aggregated_attribution_vectors[:, 0]` — CLS 행만 취함 (`modeling_bert.py:1497`) |
| 분류기(classifier) 출력 | `(B, N, C)` | `einsum("ld,bkd->bkl")`, C=클래스 수 (`modeling_bert.py:2063`) |

최종 산출물 `(B, N, C)` 가 DecompX의 결론이다: **토큰 k가 클래스 c의 로짓에 더한 부호 있는 실수 값**.
여기엔 노름(norm)이 적용되지 않는다 — 부호가 살아 있다 (`modeling_bert.py:2138`).

---

## 2. 원본 구현 구조 — 컴포넌트별 전파 규칙

### 2.1 임베딩 초기화 — 사실은 임베딩 모듈이 아니라 0번 층 안에서 생성된다

**`BertEmbeddings` 는 손대지 않았다.** `modeling_bert.py:181-240` 구간을 `decompx|Fayyaz` 로
grep 하면 히트 0건이다. 기여 벡터는 인코더 진입 시 `None` 으로 시작해서(`:887`),
0번 층 안에서 항등 대각(identity diagonal)으로 태어난다.

```python
# modeling_bert.py:772-779
hidden_shape = hidden_states.size()  # (batch, seq_length, all_head_size)
device = hidden_states.device
residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device),
                        hidden_states)  # diagonal representations (hidden states)
residual_weighted_layer = summed_weighted_layer + residual
```

수식: `A⁰[b,i,j,:] = δ_ij · h⁰[b,i,:]` — 각 토큰이 자기 자신에게 100% 귀속된 상태에서 출발.
분해의 기준점(basis)은 **임베딩 층의 출력**(단어+위치+타입 임베딩을 더하고 LayerNorm+드롭아웃까지
끝낸 것)이지 원시 토큰 id가 아니다. 즉 위치 임베딩 기여를 따로 떼어내지 **않는다**.

패딩 마스크는 여기서 적용되지 않는다. `torch.eye` 는 패딩 위치에도 자기 귀속 행을 만든다.

### 2.2 셀프 어텐션과 값(value) 투영

어텐션 확률은 **한 번만 계산하고 재사용**한다. 다시 계산하지 않는다.

```python
# modeling_bert.py:377-381
if decompx_ready:
    outputs = (context_layer, attention_probs, value_layer, decomposed_value_layer)
    return outputs
```

기여 벡터에 값 가중치(value weight)를 먹이는 곳 — **편향(bias)은 여기서 빠진다**:

```python
# modeling_bert.py:319-321
if attribution_vectors is not None:
    decomposed_value_layer = torch.einsum("bijd,vd->bijv", attribution_vectors, self.value.weight)
    decomposed_value_layer = self.transpose_for_scores_for_decomposed(decomposed_value_layer)
```

`transpose_for_scores_for_decomposed` (`:274-280`) 가 `(B,N,N,H·V) → (B,H,N,N,V)` 로 재배열한다
(H=헤드 수 12, V=헤드당 차원 64).

이어서 출력 투영 가중치를 **헤드별로 먼저 곱하고**, 어텐션 확률로 섞고, 헤드를 **합친다**:

```python
# modeling_bert.py:757-789
headmixing_weight = self.attention.output.dense.weight.view(
    self.all_head_size, self.num_attention_heads, self.attention_head_size)   # (D, H, V)

if decomposed_value_layer is None or decompx_config.aggregation != "vector":
    transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, headmixing_weight)
    weighted_layer   = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
    summed_weighted_layer = weighted_layer.sum(dim=1)      # sum over heads
    ...
    residual_weighted_layer = summed_weighted_layer + residual
    accumulated_bias = self.attention.output.dense.bias
else:
    transformed_layer = torch.einsum('bhsqv,dhv->bhsqd', decomposed_value_layer, headmixing_weight)
    weighted_layer   = torch.einsum('bhks,bhsqd->bhkqd', attention_probs, transformed_layer)
    summed_weighted_layer = weighted_layer.sum(dim=1)      # sum over heads
    residual_weighted_layer = summed_weighted_layer + attribution_vectors
    accumulated_bias = torch.matmul(self.attention.output.dense.weight,
                                    self.attention.self.value.bias) + self.attention.output.dense.bias
```

색인 뜻: `b`=배치, `h`=헤드, `k`=질의(목적지 토큰 i), `s`=키 위치, `q`=출처 토큰 j, `d`=은닉, `v`=헤드 차원.
헤드를 concat 하지 않고 **더하는** 것이 정확한 이유는 W^O 를 헤드별로 이미 곱했기 때문이다
(concat 후 한 번 곱하기 == 헤드별 곱하고 더하기).

수식: `Attn_A[b,i,j,:] = Σ_h Σ_s P[b,h,i,s] · W^O_h · (W^V A[b,s,j,:])`

값 편향 `b_v` 는 위 einsum 에 안 들어갔으므로, 0번 층 경로(값 편향이 `value_layer` 안에 이미 있음)와
합성 경로(없음)를 구분해 `accumulated_bias` 에 `W^O b_v + b_O` 로 합류시킨다. 이 구분을 놓치면
재구성 오차가 층마다 누적된다.

### 2.3 첫 번째 잔차 연결(residual)

두 부분으로 나뉜다. **(a) 순전파 쪽**: LayerNorm 이전 합을 밖으로 꺼내도록 모듈을 쪼갰다.

```python
# modeling_bert.py:398-409  (BertSelfOutput.forward)
hidden_states = self.dense(hidden_states)
hidden_states = self.dropout(hidden_states)
# hidden_states = self.LayerNorm(hidden_states + input_tensor)
pre_ln_states = hidden_states + input_tensor      # added by Fayyaz / Modarressi
post_ln_states = self.LayerNorm(pre_ln_states)    # added by Fayyaz / Modarressi
if decompx_ready:
    return post_ln_states, pre_ln_states
```

**(b) 분해 쪽**: 그냥 텐서 덧셈이다 (위 2.2 발췌의 `residual_weighted_layer = ... + residual`
또는 `... + attribution_vectors`). 덧셈은 출처 축에 대해 분배법칙이 성립하므로 **오차 0**.

### 2.4 LayerNorm 분해

```python
# modeling_bert.py:570-585
def ln_decomposer(self, attribution_vectors, pre_ln_states, gamma, beta, eps,
                  include_biases=True, bias_decomp_type="absdot"):
    mean = pre_ln_states.mean(-1, keepdim=True)                                   # (b,s,1)   m(y=Σy_j)
    var  = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True).unsqueeze(dim=2)  # (b,s,1,1) s(y)
    each_mean = attribution_vectors.mean(-1, keepdim=True)                        # (b,s,k,1) m(y_j)
    normalized_layer = torch.div(attribution_vectors - each_mean, (var + eps) ** (1 / 2))
    post_ln_layer = torch.einsum('bskd,d->bskd', normalized_layer, gamma)
    if include_biases:
        return self.bias_decomposer(beta, post_ln_layer, bias_decomp_type=bias_decomp_type)
    else:
        return post_ln_layer
```

수식: `LN_j = γ ⊙ (a_j − mean_D(a_j)) / sqrt(var_D(y) + ε)`, 여기에 β 를 따로 배분.

포팅 시 반드시 지킬 3가지:
1. **평균은 기여 벡터마다 각자** 뺀다(`each_mean`). Σ_j mean(a_j) = mean(y) 이므로 합이 보존된다.
2. **분산은 진짜 합쳐진 상태 y 에서 한 번만** 구하고 출처 축으로 브로드캐스트한다(`.unsqueeze(dim=2)`).
   즉 표준편차는 토큰당 **상수** 취급 — 이게 유일한 "선형화" 가정이다.
3. `eps` 는 제곱근 **안**에 들어간다. 분산은 편향 없는 추정이 아니라 `.mean()`(=D로 나눔) 이다.

### 2.5 FFN과 GELU 분해

```python
# modeling_bert.py:629-647
def ffn_decomposer(self, attribution_vectors, intermediate_hidden_states, intermediate_output,
                   include_biases=True, approximation_type="GeLU_LA", bias_decomp_type="absdot"):
    post_first_layer = torch.einsum("ld,bskd->bskl", self.intermediate.dense.weight, attribution_vectors)
    if include_biases:
        post_first_layer = self.bias_decomposer(self.intermediate.dense.bias, post_first_layer, ...)

    if approximation_type == "ReLU":
        mask_for_gelu_approx = (intermediate_hidden_states > 0)
        post_act_first_layer = torch.einsum("bskl, bsl->bskl", post_first_layer, mask_for_gelu_approx)  # 죽은 줄
        post_act_first_layer = post_first_layer * mask_for_gelu_approx.unsqueeze(dim=-2)
    elif approximation_type == "GeLU_LA":
        post_act_first_layer = self.gelu_decomposition(post_first_layer, ...)
    elif approximation_type == "GeLU_ZO":
        post_act_first_layer = self.gelu_zo_decomposition(post_first_layer, intermediate_hidden_states,
                                                          intermediate_output)

    post_second_layer = torch.einsum("bskl, dl->bskd", post_act_first_layer, self.output.dense.weight)
    if include_biases:
        post_second_layer = self.bias_decomposer(self.output.dense.bias, post_second_layer, ...)
    return post_second_layer
```

`l` = 중간 차원(intermediate_size 3072). 편향은 **중간 공간에서 한 번(3072차원), 은닉 공간에서
한 번(768차원)** 총 두 번 배분된다.

기본값이자 데모가 쓰는 활성 근사는 **0-원점 기울기(zero-origin slope, `GeLU_ZO`)**:

```python
# modeling_bert.py:624-627
def gelu_zo_decomposition(self, attribution_vectors, intermediate_hidden_states, intermediate_output):
    m = intermediate_output / (intermediate_hidden_states + 1e-12)
    mx = attribution_vectors * m.unsqueeze(dim=-2)
    return mx
```

수식: `m = gelu(x)/x` (원점을 지나는 직선의 기울기), 기여는 `m·a_j`.
`Σ_j m·a_j = m·x = gelu(x)` 이므로 **구조적으로 합 보존**. 절편 항이 아예 없다.

대안 `GeLU_LA` 는 접선 근사다:

```python
# modeling_bert.py:588-601
def gelu_linear_approximation(self, intermediate_hidden_states, intermediate_output):
    def phi(x):        return (1 + torch.erf(x / math.sqrt(2))) / 2.
    def normal_pdf(x): return torch.exp(-(x**2) / 2) / math.sqrt(2. * math.pi)
    def gelu_deriv(x): return phi(x) + x * normal_pdf(x)
    m = gelu_deriv(intermediate_hidden_states)
    b = intermediate_output - m * intermediate_hidden_states
    return m, b
```

절편 `b` 를 편향처럼 출처 토큰에 재배분하므로(`gelu_decomposition`, `:603-621`) 합은 여전히 보존된다.
단 `:610` 의 `cosine_similarity(mx, b)` 는 `dim=-1` 이 빠져 있어 `abssim` 조합에서 터진다.

### 2.6 두 번째 잔차 + LayerNorm

```python
# modeling_bert.py:808-831
if decompx_config.include_FFN:
    post_ffn_layer = self.ffn_decomposer_fast if decompx_config.FFN_fast_mode else self.ffn_decomposer(
        attribution_vectors=post_ln_layer,
        intermediate_hidden_states=pre_act_hidden_states,
        intermediate_output=intermediate_output,
        approximation_type=decompx_config.FFN_approx_type,
        include_biases=decompx_config.include_biases,
        bias_decomp_type=bias_decomp_type)
    pre_ln2_layer = post_ln_layer + post_ffn_layer
else:
    pre_ln2_layer = post_ln_layer
    post_ffn_layer = None

if decompx_config.include_LN2:
    post_ln2_layer = self.ln_decomposer(
        attribution_vectors=pre_ln2_layer, pre_ln_states=pre_ln2_states,
        gamma=self.output.LayerNorm.weight.data, beta=self.output.LayerNorm.bias.data,
        eps=self.output.LayerNorm.eps, ...)
```

`BertOutput.forward` (`:503-514`) 도 같은 방식으로 `pre_ln_states` 를 꺼내 준다.
LN2 는 2.4의 같은 함수를 재사용한다.

> ⚠ **`:809` 삼항 연산자 버그 (실행 확인 대신 정적 판독)**: `A if cond else B(...)` 로 파싱되어
> `FFN_fast_mode=True` 면 `post_ffn_layer` 에 **호출되지 않은 바운드 메서드 객체**가 들어가고
> `:817` 의 `텐서 + 메서드` 에서 TypeError 가 난다. 포팅 시 그대로 베끼지 말고 실제로 호출할 것.
> 기본값은 False 라 원본 실행 경로에는 영향이 없다.

### 2.7 층 간 전달 (layer-to-layer)

DecompX의 정체성. **사후에 층별 지도를 곱하는 게 아니라, 기여 벡터 자체를 다음 층에 넣는다.**

```python
# modeling_bert.py:940-943  (BertEncoder.forward 안의 층 호출)
layer_outputs = layer_module(
    hidden_states,
    aggregated_encoder_vectors,   # ← 두 번째 위치 인자 = attribution_vectors
    attention_mask, ...)

# modeling_bert.py:974-982
elif decompx_config.aggregation == "vector":
    aggregated_encoder_vectors = decompx_output.encoder[0][1]
    if decompx_config.include_classifier_w_pooler:
        decompx_output.aggregated = (aggregated_encoder_vectors,)   # 원시 텐서 그대로 (풀러가 소비)
    else:
        decompx_output.aggregated = output_builder(aggregated_encoder_vectors,
                                                   decompx_config.output_aggregated)
decompx_output.encoder = output_builder(decompx_output.encoder[0][1], decompx_config.output_encoder)
```

층 안에서 `encoder` 필드는 항상 하드코딩된 `"both"` 로 만들어지므로(`:841`)
`[0][0]`=노름, `[0][1]`=벡터다. rollout 모드는 `[0][0]`(노름) 만 쓰고 행렬 곱 사슬을 만든다(`:956-971`).

`aggregation=None` 이면 `aggregated_encoder_vectors` 가 영원히 `None` 이라 매 층이 항등 대각으로
다시 시작한다 = **층별 국소 분해**이지 DecompX 가 아니다. 우리는 `"vector"` 를 쓴다.

### 2.8 풀러(pooler)

```python
# modeling_bert.py:1029-1037  (BertPooler, pre-activation 도 함께 반환)
first_token_tensor = hidden_states[:, 0]
pre_pooled_output = self.dense(first_token_tensor)
pooled_output = self.activation(pre_pooled_output)     # nn.Tanh()
if decompx_ready:
    return pooled_output, pre_pooled_output
```

```python
# modeling_bert.py:1347-1356  (BertModel.ffn_decomposer)
post_pool = torch.einsum("ld,bsd->bsl", self.pooler.dense.weight, attribution_vectors)
if include_biases:
    post_pool = self.bias_decomposer(self.pooler.dense.bias, post_pool, bias_decomp_type=bias_decomp_type)
if tanh_approx_type == "LA":
    post_act_pool = self.tanh_la_decomposition(post_pool, pre_act_pooled, post_act_pooled, ...)
else:
    post_act_pool = self.tanh_zo_decomposition(post_pool, pre_act_pooled, post_act_pooled)
```

```python
# modeling_bert.py:1342-1345
def tanh_zo_decomposition(self, attribution_vectors, pre_act_pooled, post_act_pooled):
    m = post_act_pooled / (pre_act_pooled + 1e-12)
    mx = attribution_vectors * m.unsqueeze(dim=-2)
    return mx
```

호출부에서 **CLS 행만 잘라 4차원 → 3차원**으로 떨어진다:

```python
# modeling_bert.py:1490-1505
decompx_idx = -2 if decompx_config.output_all_layers else -1
aggregated_attribution_vectors = encoder_outputs[decompx_idx].aggregated[0]        # (B,N,N,D)
encoder_outputs[decompx_idx].aggregated = output_builder(aggregated_attribution_vectors,
                                                         decompx_config.output_aggregated)
pooler_decomposed = self.ffn_decomposer(
    attribution_vectors=aggregated_attribution_vectors[:, 0],                      # (B,N,D)  ← CLS 행
    pre_act_pooled=pre_act_pooled, post_act_pooled=pooled_output, ...)
encoder_outputs[decompx_idx].pooler = pooler_decomposed                            # 원시 텐서로 저장
```

**순서 주의**: 원시 벡터를 먼저 꺼내 쓰고 **그 다음에** 필드를 사용자 요청 형태로 덮어쓴다.
포팅할 때 순서를 바꾸면 풀러가 노름 스칼라를 먹게 된다.

### 2.9 분류기 헤드(classifier head)

```python
# modeling_bert.py:2062-2067
def ffn_decomposer(self, attribution_vectors, include_biases=True, bias_decomp_type="absdot"):
    post_classifier = torch.einsum("ld,bkd->bkl", self.classifier.weight, attribution_vectors)
    if include_biases:
        post_classifier = self.bias_decomposer(self.classifier.bias, post_classifier,
                                               bias_decomp_type=bias_decomp_type)
    return post_classifier
```

`l`=클래스, `d`=은닉, `k`=출처 토큰 → `(B, N, C)`.
분류기 편향도 버리지 않고 **로짓 공간에서** 토큰들에 배분한다. 그래서 토큰 축으로 더하면 진짜 로짓이 나온다.

```python
# modeling_bert.py:2116-2138
if decompx_config and decompx_config.include_classifier_w_pooler:
    decompx_idx = -2 if decompx_config.output_all_layers else -1
    aggregated_attribution_vectors = outputs[decompx_idx].pooler            # 원시 (B,N,D) 먼저 읽고
    outputs[decompx_idx].pooler = output_builder(aggregated_attribution_vectors,
                                                 decompx_config.output_pooler)   # 그 다음 덮어씀
    classifier_decomposed = self.ffn_decomposer(attribution_vectors=aggregated_attribution_vectors, ...)
    ...
    outputs[decompx_idx].classifier = classifier_decomposed if decompx_config.output_classifier else None
```

실제 로짓은 `pooled_output = self.dropout(pooled_output); logits = self.classifier(pooled_output)`
(`:2113-2114`) 로 나온다. 분해 경로에는 드롭아웃이 없으므로 **`model.eval()` 에서만 일치**한다.

### 2.10 편향 분해(bias decomposition)

모델의 모든 편향이 이 함수 하나를 통과한다.

```python
# modeling_bert.py:538-567 (발췌)
if bias_decomp_type == "absdot":
    weights = torch.abs(torch.einsum("bskd,d->bsk", attribution_vectors, bias))
elif bias_decomp_type == "abssim":
    weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
    weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
elif bias_decomp_type == "norm":
    weights = torch.norm(attribution_vectors, dim=-1)
elif bias_decomp_type == "equal":
    weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
elif bias_decomp_type == "cls":
    weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
    weights[:,:,0] = 1.0
elif bias_decomp_type == "dot":
    weights = torch.einsum("bskd,d->bsk", attribution_vectors, bias)
elif bias_decomp_type == "biastoken":
    ...  # 출처 축을 N → N+1 로 늘리고 마지막 칸에 편향을 그대로 쌓음, 조기 return

weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
return attribution_vectors + weighted_bias
```

가중치 합이 1이므로 **편향 총량이 정확히 보존**된다. 어느 토큰에 얼마나 줄지(누가 공을 가져갈지)만
바뀌고 총합은 안 바뀐다. 유일한 예외: 모든 가중치가 0이면 `+1e-12` 때문에 편향이 **통째로 사라진다**.

편향이 들어가는 지점 목록 (한 층당):
`W^O b_v + b_O` (`:790-793`) → LN1 β (`:583`) → `intermediate.dense.bias` (`:632`) →
`output.dense.bias` (`:645`) → LN2 β. 그리고 헤드에서 `pooler.dense.bias` (`:1350`),
`classifier.bias` (`:2065`).

### 2.11 출력 변환 — `output_builder`

```python
# modeling_bert.py:165-179
def output_builder(input_vector, output_mode):
    if output_mode is None:                 return None
    elif output_mode == "vector":           return (input_vector,)
    elif output_mode == "norm":             return (torch.norm(input_vector, dim=-1),)
    elif output_mode == "both":             return ((torch.norm(input_vector, dim=-1), input_vector),)
    elif output_mode == "distance_based":   ...
```

노름(L2)은 **여기서만** 취해진다. 노트북 후처리에는 노름 계산이 없다. 모든 반환값이 1-튜플이라
층별로 이어붙이면 길이 12 튜플이 된다.

> ⚠ **원본 배선 버그 2건 (소스 확인 완료, 우리 포팅에서는 고칠 것)**
> - `:838` `LN1=output_builder(post_ln_layer, decompx_config.output_res2)` — `output_LN1` 이어야 한다.
> - `:895` `FFN=() if decompx_config.output_LN1 else None` — `output_FFN` 이어야 한다 (`:988` 은 `output_FFN` 로 누적).
>
> 데모 설정은 여섯 개 중간 `output_*` 를 전부 `None` 으로 두기 때문에 원본 실행에서는 터지지 않는다.

---

## 3. 정확(exact) vs 근사(approximate) 판정표

"정확"의 정의: **출처 축으로 더하면 진짜 활성값이 부동소수점 오차 내에서 복원된다.**

| 컴포넌트 | 판정 | 근거 |
|---|---|---|
| 임베딩 초기화 (항등 대각) | **정확** | `δ_ij·h⁰` 이므로 정의상 합 = `h⁰` |
| 값 투영 `W^V` | **정확** | 순수 선형 einsum (`:320`) |
| 헤드별 `W^O` + 헤드 합 | **정확** | concat-후-곱 == 헤드별-곱-후-합 (`:758-762`, `:782`) |
| 어텐션 혼합 `probs @ V` | **정확 (조건부)** | 확률을 **상수**로 취급. 확률 자체는 분해하지 않으므로 재구성은 정확하지만 "귀속 의미"는 근사 |
| 잔차 1 · 잔차 2 | **정확** | 텐서 덧셈 (`:779`/`:789`, `:817`) |
| LayerNorm 1 · 2 | **정확 (조건부)** | 각자 평균 뺌 + 공유 표준편차. 표준편차를 상수로 고정한 게 유일한 가정 (`:570-585`) |
| FFN 선형 두 층 | **정확** | 순수 einsum (`:630`, `:643`) |
| GELU `GeLU_ZO` | **정확** | `m·x = gelu(x)` 항등, 절편 없음 (`:624-627`) |
| GELU `GeLU_LA` | **정확** | 접선 절편 `b` 를 재배분해 회수 (`:603-621`) — 단 `abssim` 조합은 `dim` 누락으로 crash (`:610`) |
| GELU `ReLU` 근사 | **근사(손실)** | `relu(x) ≠ gelu(x)` — 합이 진짜 활성값과 다름 (`:634-637`) |
| tanh 풀러 `ZO` / `LA` | **정확** | GELU와 같은 두 스킴 (`:1342-1345` / `:1322-1340`) |
| 편향 분해 (모든 종류) | **정확** | 가중치 합 = 1 (`:565`). 예외: 전 가중치 0이면 `1e-12` 때문에 편향 소실 |
| `include_biases=False` | **근사(의도적 손실)** | 모든 편향을 버림 |
| 풀러 dense / 분류기 dense | **정확** | 순수 einsum (`:1348`, `:2063`) |
| 드롭아웃(dropout) | **`eval()` 에서만 정확** | 분해 경로에 드롭아웃이 없음 (`:2113`) |
| rollout 집계 | **근사(다른 방법)** | 노름 공간 행렬 곱. DecompX가 아니라 GlobEnc 기준선 (`:956-971`) |

**결론**: 데모 기본 설정(`include_biases=True`, LN1/FFN/LN2 전부 True, `GeLU_ZO`, tanh `ZO`,
`aggregation="vector"`) + `eval()` + fp32 이면 **재구성은 수치 오차 수준까지 정확**하다.
이게 4~7장 검증 오라클의 근거다.

---

## 4. DeBERTaV2 대조표

기준: transformers **4.51.3**, 체크포인트 `meta-llama/Llama-Prompt-Guard-2-86M`
(hidden 768 / 12층 / 12헤드 / intermediate 3072 / `layer_norm_eps=1e-07` / `num_labels=2`).

**실행 확인한 체크포인트 사실** (실제 로드해서 출력):
`position_embeddings=None`, `token_type_embeddings=None`, `embed_proj=None`, `encoder.conv=None`,
`intermediate_act_fn=GELUActivation()`, `pooler.dropout.p=0`, `classifier dropout p=0.1`, `num_labels=2`.
config 에 **`position_biased_input: false`**, **`type_vocab_size: 0`**, `pos_att_type: ["p2c","c2p"]`,
`share_att_key: true`, `position_buckets: 256`, `norm_rel_ebd: "layer_norm"` 이 들어 있다.

| BERT 컴포넌트 | DeBERTaV2 대응 | 포팅에서 달라지는 것 |
|---|---|---|
| `BertEmbeddings` (단어+위치+타입, LN, dropout) | `DebertaV2Embeddings` (`:496-574`) | **이 체크포인트에선 위치 임베딩도 타입 임베딩도 없다.** 실효 계산은 `LayerNorm(word_emb(ids)) * mask[:,:,None]` 뿐 (`:562`, `:571`). 위치 정보는 전부 상대 어텐션으로만 들어간다. 마스크 곱은 토큰별 0/1 스칼라라 어떤 선형 분해와도 교환 가능 → 기여 벡터 전 슬라이스에 동일하게 곱하면 끝 |
| `BertSelfAttention` (절대 위치) | `DisentangledSelfAttention` (`:146-355`) | 헤드 배치가 `(B,N,D) → (B*H, N, 64)` 로 **배치와 헤드가 0번 축에 합쳐진다** (`transpose_for_scores`, `:195-198`). 모듈 이름은 `query_proj/key_proj/value_proj`. `attention_probs` 만 `(B,H,N,N)` 로 되돌려지고 `value_layer` 는 합쳐진 채로 남으므로, 분해 einsum 전에 `value.view(B,H,N,64)` 로 **명시적 복원 필요** |
| — | c2p / p2c 상대 편향 (`disentangled_attention_bias`, `:286-355`) | **점수(score)에만 더해진다** (`:260`). 값 경로에는 안 들어간다 → 5장 참조 |
| — | `share_att_key=True` | 상대 위치 질의/키를 `query_proj/key_proj` **같은 가중치**로 만든다 (`:306-312`). 분해에는 무관 — 어차피 확률 계산에만 쓰임 |
| — | `rel_embeddings` + 인코더 전용 LayerNorm | `get_rel_embedding()` 이 `self.LayerNorm(rel_embeddings.weight)` 를 forward 당 **한 번** 계산해 12층이 공유 (`:607-611`). 층별 LayerNorm과 이름이 겹치니 주의. 분해에는 무관 |
| `BertSelfOutput` | `DebertaV2SelfOutput` (`:48-59`) | 구조 동일 (`dense → dropout → LayerNorm(x + input)`). **`pre_ln_states` 반환 개조를 똑같이 넣으면 됨** |
| `BertIntermediate` | `DebertaV2Intermediate` (`:394-406`) | 동일. **`pre_act_hidden_states` 반환 개조 필요**. 활성은 `ACT2FN["gelu"] = GELUActivation` = 정확한 erf GELU (tanh 근사 아님) |
| `BertOutput` | `DebertaV2Output` (`:410-422`) | 동일 (`:421`). `pre_ln_states` 반환 개조 필요 |
| `BertLayer` | `DebertaV2Layer` (`:426-456`) | 반환이 항상 2-튜플. **`output_attentions` 인자가 마지막 위치**인데 `DebertaV2Attention`/`DisentangledSelfAttention` 에선 세 번째 위치 — 위치 인자로 호출하면 어긋난다 |
| `BertEncoder` | `DebertaV2Encoder` (`:577-703`) | 기여 벡터를 층에 넘기는 배선이 없으므로 신설. `i == 0 and self.conv is not None` 분기(`:686`)는 이 체크포인트에서 죽은 코드지만 **로드 시 assert 로 막을 것** |
| 어텐션 마스크 (덧셈형 `0 / -inf`) | `get_attention_mask` (`:613-620`) — **불리언 외적 `(B,1,N,N)`** | `masked_fill(~mask, finfo.min)` 후 softmax (`:267-269`). rollout 의 `torch.exp(attention_mask)` 트릭은 **무효** — 우리는 vector 모드만 쓰므로 무해하지만 rollout 을 포팅하려면 다시 써야 함 |
| — | 패딩 행 동작 | 외적 마스크라 **패딩 질의 행은 전부 마스킹 → softmax 가 균등 1/N**. 실측: `[0.1111]×9`. 또 인코더 안에서 은닉 상태를 다시 마스킹하지 않아 실측 마지막 층 패딩 행 노름 25.30 (0 아님). 집계 시 **명시적 마스킹 필수** |
| `BertPooler` (dense → tanh, dropout 없음) | `ContextPooler` (`:1098-1117`) | **순서가 다르다**: `hidden[:,0] → dropout → dense → ACT2FN[pooler_hidden_act]`. 그리고 활성이 **tanh 가 아니라 gelu** (`pooler_hidden_act="gelu"`). `pooler_dropout=0` 이라 드롭아웃은 항등 |
| `BertForSequenceClassification` (`self.classifier`) | `DebertaV2ForSequenceClassification` (`:1127-1236`) | 풀러가 base model 이 아니라 **헤드에 붙어 있다** (RoBERTa 배치) 지만 분류기는 별도 `nn.Linear` (BERT 배치). 헤드 3줄: `pooler → dropout(0.1) → classifier` (`:1191-1193`) |
| `DecompXConfig.tanh_approx_type` | — | **적용 불가**. 풀러 활성이 gelu 이므로 `gelu_zo_decomposition` / `gelu_decomposition` 을 재사용해야 한다. 필드 이름을 `pooler_approx_type` 으로 바꾸거나 gelu 를 의미하도록 재해석할 것 |

---

## 5. 핵심 판정 — 상대 어텐션이 분해를 깨뜨리는가

### 판정: **깨뜨리지 않는다. 구조 변경 0. 확률 값만 달라진다.**

근거는 소스 한 곳으로 끝난다. `DisentangledSelfAttention.forward` 에서 상대 위치 항 `rel_att` 가
**어디에 더해지는지** 보면 된다.

```python
# modeling_deberta_v2.py:252-281
attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
if self.relative_attention:
    rel_embeddings = self.pos_dropout(rel_embeddings)
    rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos,
                                               rel_embeddings, scale_factor)
if rel_att is not None:
    attention_scores = attention_scores + rel_att          # ← 점수에만 더해진다 (:260)
...
attention_mask = attention_mask.bool()
attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
attention_probs = nn.functional.softmax(attention_scores, dim=-1)
attention_probs = self.dropout(attention_probs)
context_layer = torch.bmm(
    attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
)                                                          # ← 값 경로엔 상대 항이 없다 (:272)
```

`disentangled_attention_bias` (`:286-355`) 전체를 읽어도 c2p 는 `bmm(query_layer, pos_key_layer.T)`,
p2c 는 `bmm(key_layer, pos_query_layer.T)` 이고, **`value_layer` 는 단 한 번도 등장하지 않는다.**
반환값 `score` 는 (헤드, 질의, 키) 당 스칼라다.

즉 값에서 문맥까지의 사상은 여전히

```
context[b,h,i,:] = Σ_k  P[b,h,i,k] · value[b,h,k,:]
```

이고, `P` 가 어떻게 만들어졌든 (내용-내용만이든, c2p/p2c 를 더했든) **`value` 에 대해 선형**이다.
DecompX 는 애초에 `P` 를 상수로 취급하므로, 상대 어텐션은 그 상수의 **숫자만** 바꾼다.
분해 규칙(`Σ_h Σ_k P·W^O_h·W^V·A`)은 한 글자도 안 바뀐다.

### 실측 증거

실제 체크포인트를 fp32 `eval()` 로 로드해 0번 층에서 `probs @ value` 를 직접 재조립하고
모듈이 내놓은 진짜 `context_layer` 와 비교했다.

```
tv 4.51.3  num_labels 2  pos_emb None  tt None  conv None
act GELUActivation()  pool_drop 0  cls_drop 0.1
logits tensor([[-5.1907,  2.5379],
               [ 1.0237e-03, -6.9897]])
emb maxdiff 0.0        # LayerNorm(word_emb(ids)) * mask  vs  hidden_states[0]
ctx maxdiff 0.0        # einsum('bhqk,bhkd->bhqd', attentions[0], value) .reshape  vs  실제 context_layer
pad row probs tensor([0.1111]*9)   mask tensor([1,1,1,0,0,0,0,0,0])
hid0 padnorm 0.0   hidLast padnorm 25.303
```

`ctx maxdiff 0.0` — 상대 어텐션이 켜진 상태(`pos_att_type=["p2c","c2p"]`)에서도
문맥 벡터는 `attention_probs @ value_layer` 로 **비트 단위로 재현된다**.
값 경로에 추가 항이 붙는다면 이 값이 0일 수 없다. 가설은 검증됐다.

### 파생 주의사항 두 개

1. **위치 기여를 따로 뽑을 수는 없다.** DecompX 는 원래도 위치 임베딩 기여를 분리하지 않지만
   (임베딩 출력에서 시작하므로), DeBERTa 는 위치 정보가 **확률 안에만** 있어서 더더욱 분리 불가능하다.
   "위치가 얼마나 기여했나" 라는 질문은 이 프레임에서 답할 수 없다 — 논문/보고에서 주장하지 말 것.
2. **패딩 질의 행이 균등 확률(1/N)을 갖는다.** 위 실측대로다. 실제 토큰 행은 패딩 키에 정확히 0을
   주므로 로짓은 오염되지 않지만, 우리가 `(B,N,N)` 전체를 평균/노름 내면 쓰레기가 섞인다.
   **집계 시 `attention_mask` 로 행·열을 모두 잘라낼 것.**

---

## 6. 포팅 계획 — `src/pg2_decompx/modeling_deberta_v2_decompx.py`

의존 순서대로. 각 단계는 앞 단계 없이는 sanity check 를 못 돌린다.

**0단계 — 로드 가드 (파일 맨 앞, 함수 하나)**
```python
def _assert_supported(config):
    assert getattr(config, "conv_kernel_size", 0) == 0          # ConvLayer 는 분해 불가
    assert config.position_biased_input is False                # 이 체크포인트 전제
    assert config.type_vocab_size == 0
    assert config.pos_att_type == ["p2c", "c2p"] and config.share_att_key
    assert config.torch_dtype in (None, "float32")              # bf16 금지
```

**1단계 — `DecompXConfig` / `DecompXOutput` 재사용**
`third_party/DecompX/src/decompx_utils.py` 를 그대로 import 한다. 새로 쓰지 않는다.
단 `tanh_approx_type` 은 풀러의 **gelu** 근사 선택자로 재해석한다 (§4 마지막 행).

**2단계 — 순수 함수 유틸 (모듈 레벨, 클래스 밖)**
원본이 `BertLayer` 메서드로 둔 것들을 자유 함수로 뽑아 풀러/분류기에서도 재사용한다.
- `output_builder(input_vector, output_mode)` — `modeling_bert.py:165-179` 그대로
- `bias_decomposer(bias, attribution_vectors, bias_decomp_type)` — 4차원 `(b,s,k,d)` 판, `:538-567`
- `bias_decomposer_3d(bias, attribution_vectors, bias_decomp_type)` — 3차원 `(b,k,d)` 판, `:1285-1312`
- `ln_decomposer(...)` — `:570-585`
- `gelu_linear_approximation` / `gelu_decomposition` / `gelu_zo_decomposition` — `:588-627`,
  **`:610` 의 `cosine_similarity` 에 `dim=-1` 추가하여 버그 수정**
- `gelu_zo_decomposition_3d` — 풀러용 `(b,k,d)` 판 (원본 `tanh_zo_decomposition` 의 gelu 버전)

**3단계 — 모듈 개조 (순전파에서 중간 상태 노출)**
전부 `decompx_ready` 플래그 추가 + 반환 확장. 로직 변경 없음.
- `DebertaV2SelfOutput.forward` → `(post_ln, pre_ln)` 반환 (`modeling_deberta_v2.py:56-58` 분리)
- `DebertaV2Intermediate.forward` → `(post_act, pre_act)` 반환
- `DebertaV2Output.forward` → `(post_ln, pre_ln)` 반환
- `ContextPooler.forward` → `(post_act, pre_act)` 반환

**4단계 — `DisentangledSelfAttentionDecompX`**
- `attribution_vectors` 를 두 번째 위치 인자로 받는다
- `decomposed_value_layer = einsum("bijd,vd->bijv", attribution_vectors, self.value_proj.weight)`
- **DeBERTa 전용 처리**: `value_layer` 를 `(B*H,N,64) → (B,H,N,64)` 로 되돌리는 헬퍼
  (`view(B, H, N, head)`); `decomposed_value_layer` 는 BERT 판 `transpose_for_scores_for_decomposed`
  와 동일하게 `(B,N,N,H,V) → permute(0,3,1,2,4)`
- `decompx_ready` 면 `(context_layer, attention_probs_4d, value_layer_4d, decomposed_value_layer)` 반환

**5단계 — `DebertaV2LayerDecompX`** (가장 큰 덩어리, `modeling_bert.py:740-843` 이식)
- `bias_decomp_type` 결정 → 어텐션 혼합 → 잔차1 → `accumulated_bias` (`W^O b_v + b_O`) 분해
  → LN1 → FFN(gelu) → 잔차2 → LN2 → `DecompXOutput` 구성
- **`FFN_fast_mode` 삼항 버그(`:809`) 는 재현하지 않는다** — 명시적 호출로 작성
- **`LN1` 을 `output_LN1` 로, all-layers `FFN` 을 `output_FFN` 로 올바로 배선** (원본 `:838`/`:895` 버그 수정)

**6단계 — `DebertaV2EncoderDecompX`**
- `aggregated_encoder_vectors` 를 층 사이로 전달 (`modeling_bert.py:940-982` 패턴)
- `output_all_layers` 누적
- **rollout 은 구현하지 않는다.** DeBERTa 의 마스크가 불리언이라 `torch.exp(attention_mask)` 가
  무효이고, 우리는 vector 모드만 필요하다. 필요해지면 그때 마스크 항만 다시 쓴다.

**7단계 — `DebertaV2ModelDecompX`**
- `encoder_outputs` 튜플에 `DecompXOutput` 을 붙인다 (`return_dict=False` 경로만)
- 임베딩 초기화는 층 0 이 처리하므로 별도 작업 없음. 단 마스크 곱(`:571`)이 은닉 상태에 이미
  반영돼 있어 항등 대각이 자동으로 마스킹된다

**8단계 — `DebertaV2ForSequenceClassificationDecompX`**
- 헤드 위치가 DeBERTa 는 `ForSequenceClassification` 에 몰려 있으므로 **풀러 분해와 분류기 분해를
  같은 프레임에서** 수행 (RoBERTa 판 레이아웃이 참고용으로 더 가깝다)
- `pooler_decomposer(attribution_vectors=aggregated[:, 0], ...)`:
  `einsum("ld,bsd->bsl", self.pooler.dense.weight, A)` → 편향 분해 → **gelu ZO 분해**
- `classifier_decomposer`: `einsum("ld,bkd->bkl", self.classifier.weight, A)` → 편향 분해
- 원시 텐서를 먼저 소비하고 나중에 `output_builder` 로 덮어쓰는 **순서 유지**

**9단계 — 테스트** (`tests/test_decompx_deberta.py`, 7장 오라클)

---

## 7. 검증 오라클 — 테스트가 확인해야 할 등식

전제: `model.eval()`, `torch.float32`, `torch.no_grad()`, 데모 설정
(`include_biases=True`, `include_LN1/FFN/LN2=True`, `FFN_approx_type="GeLU_ZO"`,
풀러 gelu ZO, `aggregation="vector"`, `include_classifier_w_pooler=True`).

기호: `A^l ∈ ℝ^{B×N×N×D}` = l번째 층 출력 기여 벡터, `h^l` = l번째 층 은닉 상태,
`P^l` = 어텐션 확률, `p` = 풀링 벡터, `z` = 로짓, `C` = 클래스 수.

**O1. 순전파 등가성 (가장 먼저 통과해야 함)**
```
our_model(input_ids, attention_mask).logits  ==  HF DebertaV2ForSequenceClassification(...).logits
허용 오차: atol=0 (fp32, 같은 연산 순서면 비트 일치해야 함)
```
개조가 순전파 수치를 바꾸지 않았음을 먼저 못 박는다. 이게 깨지면 나머지는 의미 없다.

**O2. 은닉 상태 재구성 (층별)**
```
for l in 1..12:
    A^l.sum(dim=-2)[b, i, :]  ==  hidden_states[l][b, i, :]
```
편향까지 포함해 풀어 쓰면, 한 층의 기여 벡터는 정의상

```
A^l = LN2_decomp( LN1_decomp( Σ_h Σ_k P^l[h,i,k]·W^O_h·W^V·A^{l-1}[k,·]
                              + A^{l-1}[i,·]
                              + split(W^O b_v + b_O) )
                  + FFN_decomp(·) )
```

이고, 모든 `split(·)` 의 가중치 합이 1이므로 (`modeling_bert.py:565`)

```
Σ_j A^l[b,i,j,:] = LN2( LN1( Σ_k P·W^O W^V h^{l-1}_k + h^{l-1}_i + W^O b_v + b_O )
                        + W_2·gelu(W_1·(·) + b_1) + b_2 )  =  h^l[b,i,:]
```

허용 오차: `atol=1e-5` (12층 누적 후). 층 1개만 볼 때는 `atol=1e-6`.
**패딩 위치는 비교에서 제외** (§4 참조).

**O3. 풀링 벡터 재구성**
```
pooler_decomposed.sum(dim=-2)[b, :]  ==  pooled_output[b, :]
       = gelu( W_pool · h^12[b,0,:] + b_pool )
```
`gelu ZO` 라 `m = gelu(x)/x` 가 곱해질 뿐이므로 합 보존. 허용 오차 `atol=1e-5`.

**O4. 로짓 재구성 (최종 오라클 — 가장 값싸고 가장 강력)**
```
classifier_decomposed.sum(dim=1)[b, c]  ==  logits[b, c]
       = ( W_cls · p[b,:] + b_cls )[c]
```
분류기 편향까지 토큰들에 배분되므로(`modeling_bert.py:2064-2065`) 등호가 성립한다.
허용 오차: `atol=1e-4` (로짓 크기가 O(5) 이므로 상대 오차 2e-5).

**O5. 편향 보존 (단위 테스트)**
```
bias_decomposer(bias, A, t).sum(dim=-2)  ==  A.sum(dim=-2) + bias    (모든 t 에 대해)
```
`t ∈ {absdot, abssim, norm, equal, cls, dot}`. `biastoken` 은 축이 N→N+1 로 늘어나므로 별도.
전 가중치 0 인 퇴화 입력(모든 A 가 0)에서는 **편향이 사라짐**을 명시적으로 문서화한 테스트로 남길 것.

**O6. 어텐션 확률 재사용 확인 (회귀 방지)**
```
einsum('bhqk,bhkd->bhqd', probs_returned, value_4d).reshape(B,N,D)  ==  context_layer
```
§5에서 `maxdiff 0.0` 로 실측했다. 포팅 후에도 0 이어야 한다. 0이 아니면 헤드 축 복원
(`(B*H,N,64) → (B,H,N,64)`)을 잘못 짠 것이다.

---

## 8. 미확인 / 위험 항목

- **미확인: 12층 누적 후 실제 재구성 오차 크기.** 위 허용 오차(1e-5 / 1e-4)는 연산 성격에서
  추정한 값이고 측정값이 아니다. 첫 실행에서 실측해 상수를 확정할 것. `1e-7` 같은 값을
  기대하지 말 것 — `layer_norm_eps=1e-07` 이라 LayerNorm 분모가 작아 오차가 증폭될 수 있다.
- **위험: 정밀도.** bf16/fp16 에서의 동작은 미확인. `key_layer / scale` 를 bmm **전에** 나누는
  구현(`modeling_deberta_v2.py:252`)이라 나눗셈 위치를 바꾸면 저정밀도에서 드리프트한다.
  **fp32 로 고정할 것** (프로젝트 기존 경험상 저정밀도가 평가를 망친 전례 있음).
- **위험: 패딩 행 오염.** 패딩 질의 행이 균등 1/N 확률을 갖고(실측), 인코더가 은닉 상태를 다시
  마스킹하지 않는다(실측 마지막 층 패딩 노름 25.30). 기여 행렬을 집계할 때 마스킹을 빼먹으면
  결과가 **묶음 구성(batch composition)에 따라 달라진다**. 배치 1 vs 배치 N 결과를 비교하는
  테스트를 하나 둘 것.
- **위험: 편향 배분이 패딩 토큰에도 간다.** `bias_decomposer` 의 가중치 정규화는 패딩 슬롯을
  포함한 전체 출처 축에서 이뤄진다(`:565`). 패딩이 붙은 배치와 안 붙은 배치의 실제 토큰 기여값이
  미세하게 달라진다. **미확인: 그 크기.** O4(로짓 합)는 어차피 보존되므로 테스트로는 안 잡힌다.
- **미확인: `build_rpos` 인자 뒤바뀜.** 선언은 `(query, key, relative_pos, position_buckets,
  max_relative_positions)` (`modeling_deberta_v2.py:134` 부근) 인데 호출은
  `(..., self.max_relative_positions, self.position_buckets)` 순 (`:339-345`). 자기 어텐션에서는
  `query_states=None` 이라 무해하지만, 이걸 "고치면" 상대 위치가 달라진다. **손대지 말 것.**
- **위험: DecompX 원본의 배선 버그 2건.** `:838`(LN1↔output_res2), `:895`(FFN↔output_LN1).
  데모 설정에서는 발화하지 않지만 중간 출력을 켜는 순간 `TypeError` 또는 조용한 오배선이 된다.
  우리 포팅에서는 고쳐 넣고, **고쳤다는 사실을 코드 주석에 남길 것** (원본과 비교할 사람이 헷갈린다).
- **위험: `FFN_fast_mode=True` 는 원본에서 죽은 코드.** `:809` 삼항 연산자 우선순위 문제
  (정적 판독으로 확인, 실행 미검증). 포팅에서는 명시 호출로 바꾸되, `GeLU_LA` 분기가 없어
  `theta` 가 미정의라는 점도 함께 고려할 것.
- **미확인: `include_bias_token=True` 경로.** 출처 축이 N→N+1 로 늘고 `:562` 에서 텐서를
  제자리 변형(in-place)한다. 우리 계획에는 필요 없으니 **1차 포팅에서 제외**한다.
- **버전 고정 필요.** DecompX 원본은 transformers 4.17~4.18 을 전제로 쓰였고 우리는 4.51.3 을
  쓴다. 원본 파일을 그대로 import 해 돌릴 수는 없다 (4.17 시절 `modeling_utils` 레이아웃 의존).
  → **원본을 실행 비교 기준으로 쓰려면 별도 가상환경에 4.18 을 깔아야 한다.** 미확인: 그럴 가치가
  있는지. O1~O4 는 원본 없이도 HF DeBERTa 만으로 검증 가능하므로, 1차에서는 원본 실행 비교를 생략한다.
- **`ContextPooler` 의 활성이 gelu 인 점을 놓치기 쉽다.** BERT 판을 그대로 베끼면 tanh 분해를
  적용하게 되고, O3/O4 가 조용히 틀린 채로 통과할 수 있다(둘 다 ZO 형태라 형태는 같다).
  포팅 시 `ACT2FN[config.pooler_hidden_act]` 를 실제로 읽어 분기할 것.

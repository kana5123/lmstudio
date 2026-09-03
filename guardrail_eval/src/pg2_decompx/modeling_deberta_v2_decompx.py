"""DecompX 를 DeBERTaV2(= Llama-Prompt-Guard-2-86M) 에 포팅.

설치된 transformers 패키지는 **건드리지 않는다**.  대신 이미 로드된
DebertaV2ForSequenceClassification 객체를 감싸서, 그 하위 모듈(가중치)과
원본 헬퍼(disentangled_attention_bias, build_relative_position)를 **그대로 호출**하며
순전파를 다시 엮는다.  분해 규칙은 원본 DecompX(third_party/DecompX/src/modeling_bert.py)
에서 그대로 옮겼고, 옮긴 지점마다 원본 줄 번호를 주석에 남겼다.

핵심 판정 — 상대위치 주의집중(relative attention)은 분해 구조를 바꾸지 않는다.
  modeling_deberta_v2.py:288  context_layer = torch.bmm(attention_probs, value_layer)
  p2c / c2p 항은 modeling_deberta_v2.py:250-256 에서 attention_scores 에만 더해진다.
  DecompX 는 주의집중 확률(attention_probs)을 **상수**로 취급하므로, 상대위치는
  가중치의 '값'만 바꿀 뿐 '어떤 토큰이 어떤 경로로 기여하는가'라는 분해 구조에는
  개입하지 않는다.  따라서 BERT 판 전파식을 그대로 쓸 수 있다.

분해 텐서(attribution tensor) 축 규약 — 원본과 동일:
    attrib[b, i, j, :]  = 입력 토큰 j 가 현재 층의 토큰 i 표현에 기여한 벡터
    모양 (B, N, N, H) = (배치, 대상토큰, 원천토큰, 은닉차원)
    불변식:  attrib[b, i].sum(dim=0) == hidden_states[b, i]
"""
import math
from typing import Optional

import torch
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    build_relative_position, scaled_size_sqrt,
)

from .decompx_utils import DecompXConfig, DecompXOutput, output_builder


# ---------------------------------------------------------------- 편향 분해
def bias_decomposer(bias, attrib, kind="absdot"):
    """편향 벡터 하나를 원천토큰들에게 나눠 준다.

    원본 modeling_bert.py:538-557 그대로.
    attrib: (..., N_src, H),  bias: (H,)   ->  반환 (..., N_src, H)
    가중치를 정규화해 더하므로 **합은 정확히 bias 만큼 늘어난다**(분해 총합 보존).
    """
    if kind == "absdot":
        w = torch.abs(torch.einsum("...kd,d->...k", attrib, bias))
    elif kind == "abssim":
        w = torch.abs(torch.nn.functional.cosine_similarity(attrib, bias, dim=-1))
        w = (torch.norm(attrib, dim=-1) != 0) * w
    elif kind == "norm":
        w = torch.norm(attrib, dim=-1)
    elif kind == "equal":
        w = (torch.norm(attrib, dim=-1) != 0) * 1.0
    elif kind == "cls":
        w = torch.zeros(attrib.shape[:-1], device=attrib.device, dtype=attrib.dtype)
        w[..., 0] = 1.0
    elif kind == "dot":
        w = torch.einsum("...kd,d->...k", attrib, bias)
    else:
        raise ValueError(kind)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
    return attrib + w.unsqueeze(-1) * bias


def vec_bias_decomposer(bias_vec, attrib, kind="absdot"):
    """토큰마다 다른 편향(b: (..., H))을 나눠 준다.  원본 gelu_decomposition 의 b 처리
    (modeling_bert.py:693-710) 와 같은 규칙."""
    if kind == "absdot":
        w = torch.abs(torch.einsum("...kd,...d->...k", attrib, bias_vec))
    elif kind == "norm":
        w = torch.norm(attrib, dim=-1)
    elif kind == "equal":
        w = (torch.norm(attrib, dim=-1) != 0) * 1.0
    elif kind == "cls":
        w = torch.zeros(attrib.shape[:-1], device=attrib.device, dtype=attrib.dtype)
        w[..., 0] = 1.0
    else:
        raise ValueError(kind)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
    return attrib + w.unsqueeze(-1) * bias_vec.unsqueeze(-2)


# ---------------------------------------------------------------- LayerNorm
def ln_decomposer(attrib, pre_ln_states, gamma, beta, eps,
                  include_biases=True, kind="absdot"):
    """층 정규화(LayerNorm) 분해.  원본 modeling_bert.py:560-576 그대로.

    LN(y) = (y - mean(y)) / sqrt(var(y)+eps) * gamma + beta 에서
    분산(var)은 **실제 순전파 값에서 계산해 상수로 고정**한다.  y = Σ_j y_j 이므로
    mean(y) = Σ_j mean(y_j) 이고, 상수 나눗셈은 선형이라 분해 총합이 보존된다.
      attrib: (B, N, N, H),  pre_ln_states: (B, N, H)
    """
    mean = pre_ln_states.mean(-1, keepdim=True)                       # (B,N,1)
    var = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True).unsqueeze(2)   # (B,N,1,1)
    each_mean = attrib.mean(-1, keepdim=True)                         # (B,N,N,1)
    out = (attrib - each_mean) / (var + eps).sqrt() * gamma
    if include_biases:
        out = bias_decomposer(beta, out, kind)
    return out


# ---------------------------------------------------------------- 활성함수
def act_zo(attrib, pre_act, post_act):
    """영점 기울기 근사(zero-origin).  원본 modeling_bert.py:624-627.
    m = f(x)/x 로 두고 각 기여를 같은 비율로 줄인다.  Σ_k m*a_k = m*x = f(x) 로 총합 보존."""
    m = post_act / (pre_act + 1e-12)
    return attrib * m.unsqueeze(-2)


def act_la(attrib, pre_act, post_act, deriv, kind="absdot"):
    """1차 선형 근사(linear approximation).  원본 modeling_bert.py:579-590, 693-712.
    f(x) ≈ m·x + b,  m = f'(x),  b = f(x) - m·x.  b 는 가중 분배하므로 총합 보존."""
    m = deriv(pre_act)
    b = post_act - m * pre_act
    mx = attrib * m.unsqueeze(-2)
    return vec_bias_decomposer(b, mx, kind)


def _gelu_deriv(x):
    phi = (1 + torch.erf(x / math.sqrt(2))) / 2.0
    pdf = torch.exp(-(x ** 2) / 2) / math.sqrt(2.0 * math.pi)
    return phi + x * pdf


def _tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2.0


DERIVS = {"gelu": _gelu_deriv, "gelu_new": _gelu_deriv, "tanh": _tanh_deriv}


def act_decompose(attrib, pre_act, post_act, act_name, approx="ZO", kind="absdot"):
    """approx 는 'ZO'|'LA'|'ReLU' 또는 원본 표기 'GeLU_ZO'|'GeLU_LA' 를 모두 받는다."""
    approx = {"GeLU_ZO": "ZO", "GeLU_LA": "LA"}.get(approx, approx)
    if approx == "ZO":
        return act_zo(attrib, pre_act, post_act)
    if approx == "ReLU":
        return attrib * (pre_act > 0).to(attrib.dtype).unsqueeze(-2)
    if approx == "LA":
        d = DERIVS.get(act_name)
        if d is None:
            raise ValueError(f"LA 근사에 필요한 도함수 없음: {act_name}")
        return act_la(attrib, pre_act, post_act, d, kind)
    raise ValueError(approx)


# ---------------------------------------------------------------- 래퍼 본체
class DecompXDebertaV2:
    """이미 로드된 DebertaV2ForSequenceClassification 을 감싼다.  가중치는 공유(동결)."""

    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.deberta = model.deberta
        self.encoder = model.deberta.encoder
        self.embeddings = model.deberta.embeddings
        self.pooler = model.pooler
        self.classifier = model.classifier
        self.n_heads = self.cfg.num_attention_heads
        self.head_dim = self.cfg.hidden_size // self.n_heads

    # ------------------------------------------------ 원본 주의집중 재현
    def _attention(self, attn, hidden_states, attention_mask, relative_pos, rel_embeddings):
        """DisentangledSelfAttention.forward(modeling_deberta_v2.py:200-292) 를 그대로 재현하되
        분해에 필요한 attention_probs / value_layer 를 같이 돌려준다.
        상대위치 항은 원본 메서드 disentangled_attention_bias 를 **그대로 호출**한다."""
        q = attn.transpose_for_scores(attn.query_proj(hidden_states), attn.num_attention_heads)
        k = attn.transpose_for_scores(attn.key_proj(hidden_states), attn.num_attention_heads)
        v = attn.transpose_for_scores(attn.value_proj(hidden_states), attn.num_attention_heads)

        scale_factor = 1 + ("c2p" in attn.pos_att_type) + ("p2c" in attn.pos_att_type)
        scale = scaled_size_sqrt(q, scale_factor)
        scores = torch.bmm(q, k.transpose(-1, -2) / scale.to(dtype=q.dtype))
        if attn.relative_attention:
            rel_e = attn.pos_dropout(rel_embeddings)
            scores = scores + attn.disentangled_attention_bias(q, k, relative_pos, rel_e, scale_factor)

        scores = scores.view(-1, attn.num_attention_heads, scores.size(-2), scores.size(-1))
        mask = attention_mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(q.dtype).min)
        probs = torch.nn.functional.softmax(scores, dim=-1)      # (B, h, N, N)
        probs = attn.dropout(probs)

        ctx = torch.bmm(probs.view(-1, probs.size(-2), probs.size(-1)), v)
        ctx = (ctx.view(-1, attn.num_attention_heads, ctx.size(-2), ctx.size(-1))
                  .permute(0, 2, 1, 3).contiguous())
        ctx = ctx.view(ctx.size()[:-2] + (-1,))                  # (B, N, H)

        B, N = probs.shape[0], probs.shape[-1]
        v_h = v.view(B, attn.num_attention_heads, N, -1)         # (B, h, N, hd)
        return ctx, probs, v_h

    # ------------------------------------------------ FFN 분해 (메모리 청크)
    def _ffn_decompose(self, layer, attrib, pre_act, inter_out, dcfg):
        """원본 ffn_decomposer(modeling_bert.py:629-647).
        중간차원(3072)으로 부풀리는 텐서가 (B,N,N,3072) 라 메모리를 잡아먹으므로
        대상토큰 축(N)을 ffn_chunk 개씩 끊어 계산한다.  결과는 수학적으로 동일."""
        W1, b1 = layer.intermediate.dense.weight, layer.intermediate.dense.bias
        W2, b2 = layer.output.dense.weight, layer.output.dense.bias
        kind = dcfg.bias_decomp_type
        act_name = self.cfg.hidden_act
        outs = []
        for s in range(0, attrib.shape[1], dcfg.ffn_chunk):
            a = attrib[:, s:s + dcfg.ffn_chunk]                        # (B,c,N,H)
            h = torch.einsum("ld,bskd->bskl", W1, a)                   # (B,c,N,I)
            if dcfg.include_biases:
                h = bias_decomposer(b1, h, kind)
            h = act_decompose(h, pre_act[:, s:s + dcfg.ffn_chunk],
                              inter_out[:, s:s + dcfg.ffn_chunk],
                              act_name, dcfg.FFN_approx_type, kind)
            o = torch.einsum("bskl,dl->bskd", h, W2)                   # (B,c,N,H)
            if dcfg.include_biases:
                o = bias_decomposer(b2, o, kind)
            outs.append(o)
            del a, h, o
        return torch.cat(outs, dim=1)

    # ------------------------------------------------ 한 층
    def _layer(self, layer, hidden, attrib, attention_mask, relative_pos, rel_emb, dcfg):
        attn = layer.attention.self
        ctx, probs, v_h = self._attention(attn, hidden, attention_mask, relative_pos, rel_emb)

        # --- 순전파(원본과 동일 순서) ---
        pre_ln1 = layer.attention.output.dense(ctx) + hidden
        att_out = layer.attention.output.LayerNorm(pre_ln1)
        pre_act = layer.intermediate.dense(att_out)
        inter_out = layer.intermediate.intermediate_act_fn(pre_act)
        pre_ln2 = layer.output.dense(inter_out) + att_out
        layer_out = layer.output.LayerNorm(pre_ln2)

        # --- 분해: 주의집중 + 첫 잔차 (원본 modeling_bert.py:759-790) ---
        Wo = layer.attention.output.dense.weight                       # (H, H)
        mix = Wo.view(self.n_heads * self.head_dim, self.n_heads, self.head_dim)
        # 원본은 헤드 혼합(W^O)을 **먼저** 적용해 (B,h,N,N,768) 짜리 중간 텐서를 만든다
        # (modeling_bert.py:764-767, 786-787).  N=512 면 그것만 9.6 GiB 다.
        # 수학적으로 동일하되 **헤드 합을 마지막에** 하도록 순서를 바꾸면 중간 텐서가
        # (B,h,N,N,64) 로 줄어 12배 작아진다:
        #   Σ_h Σ_v mix[d,h,v] · (Σ_s probs[h,k,s]·dv[h,s,q,v])
        if attrib is None:
            # 첫 층: 들어온 분해가 없으므로 은닉상태를 자기 자신에게 대각으로 귀속
            A = probs.unsqueeze(-1) * v_h.unsqueeze(2)                 # (B,h,N,N,hd)
            summed = torch.einsum("bhksv,dhv->bksd", A, mix)           # 헤드 합 포함
            del A
            N = hidden.shape[1]
            residual = torch.einsum("sk,bsd->bskd",
                                    torch.eye(N, device=hidden.device, dtype=hidden.dtype), hidden)
            res_w = summed + residual
            acc_bias = layer.attention.output.dense.bias               # V 편향은 v_h 안에 이미 포함
        else:
            dv = torch.einsum("bijd,vd->bijv", attrib, attn.value_proj.weight)
            B, N = dv.shape[0], dv.shape[1]
            dv = dv.view(B, N, N, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,h,N,N,hd)
            A = torch.einsum("bhks,bhsqv->bhkqv", probs, dv)           # (B,h,N,N,hd)
            del dv
            summed = torch.einsum("bhkqv,dhv->bkqd", A, mix)
            del A
            res_w = summed + attrib
            acc_bias = (torch.matmul(Wo, attn.value_proj.bias)
                        + layer.attention.output.dense.bias)
        if dcfg.include_biases:
            res_w = bias_decomposer(acc_bias, res_w, dcfg.bias_decomp_type)

        # --- 분해: LN1 -> FFN -> 잔차2 -> LN2 ---
        a = ln_decomposer(res_w, pre_ln1,
                          layer.attention.output.LayerNorm.weight.data,
                          layer.attention.output.LayerNorm.bias.data,
                          layer.attention.output.LayerNorm.eps,
                          dcfg.include_biases, dcfg.bias_decomp_type) if dcfg.include_LN1 else res_w
        del res_w
        if dcfg.include_FFN:
            a = a + self._ffn_decompose(layer, a, pre_act, inter_out, dcfg)
        if dcfg.include_LN2:
            a = ln_decomposer(a, pre_ln2,
                              layer.output.LayerNorm.weight.data,
                              layer.output.LayerNorm.bias.data,
                              layer.output.LayerNorm.eps,
                              dcfg.include_biases, dcfg.bias_decomp_type)
        return layer_out, a

    # ------------------------------------------------ 전체 순전파
    @torch.no_grad()
    def forward(self, input_ids, attention_mask, decompx_config: Optional[DecompXConfig] = None,
                output_hidden_states=False):
        """decompx_config=None 이면 분해를 끄고 원본과 동일한 순전파만 한다."""
        assert not self.model.training, "평가 모드에서만 쓴다(드롭아웃이 분해를 깨뜨림)"
        emb = self.embeddings(input_ids=input_ids, mask=attention_mask)
        att_mask = self.encoder.get_attention_mask(attention_mask)
        rel_pos = self.encoder.get_rel_pos(emb)
        rel_emb = self.encoder.get_rel_embedding()

        hidden, attrib = emb, None
        hs = [emb] if output_hidden_states else None
        per_layer = []
        for layer in self.encoder.layer:
            if decompx_config is None:
                hidden = layer(hidden, att_mask, query_states=None,
                               relative_pos=rel_pos, rel_embeddings=rel_emb)[0]
            else:
                hidden, attrib = self._layer(layer, hidden, attrib, att_mask,
                                             rel_pos, rel_emb, decompx_config)
                if decompx_config.output_all_layers:
                    per_layer.append(attrib[:, 0].clone())     # CLS 행만 보관 (B,N,H)
            if output_hidden_states:
                hs.append(hidden)

        pooled_pre = self.pooler.dense(self.pooler.dropout(hidden[:, 0]))
        pooled = ACT2FN[self.cfg.pooler_hidden_act](pooled_pre)
        logits = self.classifier(self.model.dropout(pooled))

        out = DecompXOutput()
        if decompx_config is not None:
            if decompx_config.output_all_layers and per_layer:
                out.cls_encoder = torch.stack(per_layer, dim=1)        # (B, L, N, H)
            out.encoder = output_builder(attrib, decompx_config.output_encoder)
            if decompx_config.include_classifier_w_pooler:
                # pooler: CLS 행만 사용 (원본 modeling_bert.py:1498)
                p = torch.einsum("ld,bkd->bkl", self.pooler.dense.weight, attrib[:, 0])
                if decompx_config.include_biases:
                    p = bias_decomposer(self.pooler.dense.bias, p, decompx_config.bias_decomp_type)
                # p:(B,N,H), pooled_pre/pooled:(B,H) -> 활성함수 분해가 원천토큰 축에 방송된다
                p = act_decompose(p, pooled_pre, pooled, self.cfg.pooler_hidden_act,
                                  decompx_config.act_approx_type, decompx_config.bias_decomp_type)
                out.pooler = p                                          # (B, N, H)
                if decompx_config.output_classifier:
                    c = torch.einsum("ld,bkd->bkl", self.classifier.weight, p)
                    if decompx_config.include_biases:
                        c = bias_decomposer(self.classifier.bias, c, decompx_config.bias_decomp_type)
                    out.classifier = c                                  # (B, N, C)
        return logits, hidden, hs, out

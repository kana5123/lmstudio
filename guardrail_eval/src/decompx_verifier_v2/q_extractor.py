"""MarginSensitivityExtractor: q_l = d(z_attack - z_benign) / d h_CLS^(l).

이건 'layer l 의 예측'이 아니다.  중간 층에 분류기를 붙이지 않는다.
최종 분류기는 마지막에 단 한 번만 작동하고, q_l 은 그 최종 margin 이
layer l 의 CLS 표현의 어떤 방향 변화에 민감한지를 나타낸다.
정확한 용어: layer-wise final-margin sensitivity.
"""
import torch


class MarginSensitivityExtractor:
    def __init__(self, adapter):
        self.a = adapter
        self.cls = adapter.get_decision_token_index()
        self.L = adapter.get_num_layers()

    @property
    def dest_layers(self):
        """§10: D_(L(l-1)->L(l)) 의 Query 는 목적지 층 l 의 q_l 이다.
        D 가 L1->L2 부터 시작하므로 목적지 층은 2..L 이고 K = L-1 개다."""
        return list(range(2, self.L + 1))

    def extract(self, input_ids, attention_mask, need_graph_free=True):
        """-> dict(q [B,K,d], logits [B,2], margin [B], hidden tuple(L+1) of [B,T,d])

        no_grad 로 감싸지 않는다.  base 가중치는 optimizer 에 들어가지 않으므로
        갱신되지 않지만, activation 그래프는 있어야 q 를 얻을 수 있다.
        """
        embeds = self.a.make_grad_enabled_inputs(input_ids)
        logits, hidden = self.a.forward(inputs_embeds=embeds, attention_mask=attention_mask,
                                        output_hidden_states=True)
        margin = self.a.margin(logits)                      # [B]
        dest = [hidden[l] for l in self.dest_layers]        # 11 개 [B,T,d]
        grads = torch.autograd.grad(margin.sum(), dest, create_graph=False,
                                    retain_graph=not need_graph_free, allow_unused=False)
        q = torch.stack([g[:, self.cls, :] for g in grads], dim=1)   # [B,K,d]
        return dict(q=q.detach(), logits=logits.detach(), margin=margin.detach(),
                    hidden=tuple(h.detach() for h in hidden), embeds_grad_ok=embeds.requires_grad)

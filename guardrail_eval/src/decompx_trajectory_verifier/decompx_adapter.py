"""DecompXAdapter (§34).  기존 DeBERTaV2 포트를 import 해서 쓴다.  새로 작성하지 않는다.

제공하는 두 가지 evidence:
  get_layerwise_cls_decomposition() -> C [B,L,T,d]
      논문 Eq.(1) 의 layer representation 에 대한 source-token vector decomposition 중
      CLS 행.  포트의 out.cls_encoder 가 그것이다.
  get_class_logit_decomposition() -> Y [B,T,C]
      논문 Eq.(13)-(14) 대로 pooler+classifier 까지 전파한 class 별 token 기여.
      포트의 out.classifier 가 그것이다.
"""
import torch

from src.pg2_decompx.decompx_utils import DecompXConfig
from src.pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

# 헤드까지 전파해야 하므로 include_classifier_w_pooler=True, output_classifier=True
DCFG = DecompXConfig(include_biases=True, bias_decomp_type="absdot",
                     include_classifier_w_pooler=True, output_classifier=True,
                     output_all_layers=True, output_encoder=None, output_pooler=None,
                     act_approx_type="ZO", aggregation="vector")


class DecompXAdapter:
    def __init__(self, base_adapter, dcfg=DCFG):
        self.a = base_adapter
        self.dx = DecompXDebertaV2(base_adapter.model)
        self.dcfg = dcfg
        self.L = base_adapter.get_num_layers()

    @property
    def bias_decomp_mode(self):
        return self.dcfg.bias_decomp_type

    @torch.no_grad()
    def extract(self, input_ids, attention_mask):
        """-> dict(C [B,L,T,d], Y [B,T,C], a [B,T], logits [B,C], hidden tuple(L+1))"""
        logits, _, hs, out = self.dx.forward(input_ids, attention_mask, self.dcfg,
                                             output_hidden_states=True)
        C = out.cls_encoder                                   # [B,L,T,d]
        Y = out.classifier                                    # [B,T,C]
        assert C is not None and Y is not None, "포트가 C/Y 를 내주지 않았다"
        assert C.shape[1] == self.L, f"C 층수 {C.shape[1]} != {self.L}"
        a = Y[..., self.a.get_attack_label_id()] - Y[..., self.a.get_benign_label_id()]
        return dict(C=C, Y=Y, a=a, logits=logits, hidden=hs)

    def get_layerwise_cls_decomposition(self, input_ids, attention_mask):
        return self.extract(input_ids, attention_mask)["C"]

    def get_class_logit_decomposition(self, input_ids, attention_mask):
        return self.extract(input_ids, attention_mask)["Y"]

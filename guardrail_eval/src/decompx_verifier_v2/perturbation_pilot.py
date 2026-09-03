"""§26: q_l^T D_lk 가 실제로 1차 final-margin relevance 로 동작하는지 확인.

목적지 층 l 의 CLS 에 eta * D_lk 를 더하고 downstream 을 다시 돌려
실제 margin 변화를 재서, 1차 예측 eta * q_l^T D_lk 와 비교한다.
아키텍처 검증용이며 별도 연구 분석으로 확장하지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier_v2.base_adapter import PromptGuard2Adapter
from src.decompx_verifier_v2.config import BASE_MODEL, RES
from src.decompx_verifier_v2.decompx_features import DecompXTransitionExtractor
from src.decompx_verifier_v2.q_extractor import MarginSensitivityExtractor
from transformers.activations import ACT2FN


class DownstreamRunner:
    """층 l 의 은닉상태에서 시작해 끝까지 돌려 margin 을 낸다."""

    def __init__(self, adapter):
        self.a = adapter
        self.m = adapter.model
        self.enc = adapter.model.deberta.encoder

    @torch.no_grad()
    def margin_from(self, hidden_l, layer_idx, attention_mask):
        """hidden_l: [B,T,d] = 층 layer_idx 의 출력.  layer_idx+1 부터 끝까지 실행."""
        att = self.enc.get_attention_mask(attention_mask)
        rel_pos = self.enc.get_rel_pos(hidden_l)
        rel_emb = self.enc.get_rel_embedding()
        h = hidden_l
        for layer in self.enc.layer[layer_idx:]:
            h = layer(h, att, query_states=None, relative_pos=rel_pos, rel_embeddings=rel_emb)[0]
        pooled = ACT2FN[self.m.config.pooler_hidden_act](
            self.m.pooler.dense(h[:, self.a.get_decision_token_index()]))
        logits = self.m.classifier(pooled)
        return self.a.margin(logits)


def main(n=4, etas=(1e-4, 1e-3, 1e-2), n_tokens=6, device="cuda", dtype=torch.float32):
    ad = PromptGuard2Adapter(BASE_MODEL, device, dtype=dtype)
    qx, dx, run = MarginSensitivityExtractor(ad), DecompXTransitionExtractor(ad), DownstreamRunner(ad)
    d = pd.read_parquet(Path(__file__).resolve().parents[2]
                        / "data/decompx_verifier/pg2_predictions.parquet",
                        columns=["sample_id", "text", "use", "length_ok", "token_length",
                                 "confusion_cell"])
    d = d[(d.use == "MAIN") & d.length_ok & d.token_length.between(20, 120)]
    pick = pd.concat([d[d.confusion_cell == c].head(n // 4 or 1) for c in ("TP", "FP", "TN", "FN")])
    rows = []
    for _, r in pick.iterrows():
        enc = ad.encode([r.text])
        ii, am = enc["input_ids"], enc["attention_mask"]
        qo, do = qx.extract(ii, am), dx.extract(ii, am)
        T = int(am.sum())
        m0 = run.margin_from(do["hidden"][1], 1, am)          # 검증: 층1에서 재시작해도 같은가
        assert (m0 - qo["margin"]).abs().max() < 1e-3, "downstream 재실행이 원 margin 과 다르다"
        for j in [0, 5, dx.K - 1]:                            # 얕은/중간/깊은 transition
            dest = j + 2                                      # 목적지 층
            h_dest = do["hidden"][dest]                       # [1,T,d]
            base_m = float(run.margin_from(h_dest, dest, am)[0])
            # |a| 가 큰 토큰 위주로 고른다
            a_tok = torch.einsum("d,td->t", qo["q"][0, j], do["D"][0, j, :T])
            idx = a_tok.abs().topk(min(n_tokens, T)).indices.tolist()
            for k in idx:
                Dk = do["D"][0, j, k]                          # [d]
                pred_unit = float(torch.dot(qo["q"][0, j], Dk))
                for eta in etas:
                    hp = h_dest.clone()
                    hp[0, ad.get_decision_token_index()] += eta * Dk
                    actual = float(run.margin_from(hp, dest, am)[0]) - base_m
                    rows.append(dict(sample_id=r.sample_id, cell=r.confusion_cell,
                                     transition=f"L{j+1}->L{j+2}", dest_layer=dest, token=k,
                                     eta=eta, delta_pred=eta * pred_unit, delta_actual=actual))
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    tag = "f64" if dtype == torch.float64 else "f32"
    df.to_csv(RES / f"first_order_pilot_{tag}.csv", index=False)
    print(f"[{tag}] 표본 {len(pick)}개, 측정 {len(df)}건\n")
    print("eta 별 1차 예측 대 실제 margin 변화")
    for eta, g in df.groupby("eta"):
        rel = ((g.delta_actual - g.delta_pred).abs() / (g.delta_pred.abs() + 1e-12))
        cor = np.corrcoef(g.delta_pred, g.delta_actual)[0, 1]
        print(f"  eta={eta:<7} n={len(g):>3}  상관 {cor:.6f}  "
              f"상대편차 중앙값 {rel.median():.3e}  최대 {rel.max():.3e}")
    print("\n층별 상관 (eta=1e-4)")
    sub = df[df.eta == min(etas)]
    for t, g in sub.groupby("transition"):
        print(f"  {t:<10} 상관 {np.corrcoef(g.delta_pred, g.delta_actual)[0,1]:.6f}  n={len(g)}")
    return df


if __name__ == "__main__":
    import sys as _s
    main(dtype=torch.float64 if "--f64" in _s.argv else torch.float32)

"""STEP 3: q 추출 검증 (§25 A-F)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier_v2.base_adapter import PromptGuard2Adapter
from src.decompx_verifier_v2.config import BASE_MODEL, TOL_LOGIT_MATCH
from src.decompx_verifier_v2.q_extractor import MarginSensitivityExtractor


def main(n=16, device="cuda"):
    ad = PromptGuard2Adapter(BASE_MODEL, device)
    ex = MarginSensitivityExtractor(ad)
    d = pd.read_parquet(Path(__file__).resolve().parents[2]
                        / "data/decompx_verifier/pg2_predictions.parquet",
                        columns=["text", "use", "length_ok", "token_length", "p_unsafe"])
    d = d[(d.use == "MAIN") & d.length_ok].sort_values("token_length")
    pick = pd.concat([d.head(n // 2), d[d.token_length.between(100, 300)].head(n // 2)])
    enc = ad.encode(pick.text.tolist())
    print(f"표본 {len(pick)}개, 배치 {tuple(enc['input_ids'].shape)}")
    print(f"attack_label_id={ad.get_attack_label_id()}  benign_label_id={ad.get_benign_label_id()}"
          f"  CLS index={ad.get_decision_token_index()}  L={ad.get_num_layers()}  d={ad.get_hidden_size()}")
    print(f"목적지 층 (q 대상): {ex.dest_layers}  -> K={len(ex.dest_layers)}\n")

    # --- A. inputs_embeds 경로와 input_ids 경로의 로짓이 같은가 -------------
    with torch.no_grad():
        ref = ad.forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         output_hidden_states=False)[0]
    out = ex.extract(enc["input_ids"], enc["attention_mask"])
    diff = (out["logits"] - ref).abs().max().item()
    print(f"A. 로짓 일치 (inputs_embeds vs input_ids): 최대차 {diff:.3e}  허용 {TOL_LOGIT_MATCH:.0e}"
          f"  -> {'통과' if diff <= TOL_LOGIT_MATCH else '★실패'}")

    # --- B. shape ----------------------------------------------------------
    B, K, dd = out["q"].shape
    ok_b = (B == len(pick)) and (K == len(ex.dest_layers)) and (dd == ad.get_hidden_size())
    print(f"B. q shape = {tuple(out['q'].shape)}  기대 ({len(pick)}, {len(ex.dest_layers)}, "
          f"{ad.get_hidden_size()})  -> {'통과' if ok_b else '★실패'}")

    # --- C. NaN/Inf --------------------------------------------------------
    bad = (~torch.isfinite(out["q"])).sum().item()
    print(f"C. q 의 NaN/Inf 개수 = {bad}  -> {'통과' if bad == 0 else '★실패'}")

    # --- D. 모든 목적지 층에서 그래프에 실제로 연결됐는가 --------------------
    nrm = out["q"].norm(dim=-1)                     # [B,K]
    zero_layers = (nrm.max(0).values == 0).sum().item()
    print(f"D. 층별 ||q|| 최소~최대 = {nrm.min().item():.4e} ~ {nrm.max().item():.4e}, "
          f"전부 0 인 층 {zero_layers}개 -> {'통과' if zero_layers == 0 else '★실패'}")
    print("   층별 ||q|| 평균:", " ".join(f"L{l}:{v:.3f}" for l, v in
                                          zip(ex.dest_layers, nrm.mean(0).tolist())))

    # --- E. base 파라미터 gradient 가 생기지 않았는가 -----------------------
    pg = [p.grad is not None for p in ad.model.parameters()]
    print(f"E. base 파라미터 중 grad 가 채워진 것 = {sum(pg)}개 -> {'통과' if sum(pg) == 0 else '★실패'}")

    # --- F. margin 부호 규약 ------------------------------------------------
    lg = out["logits"]
    m_manual = lg[:, ad.get_attack_label_id()] - lg[:, ad.get_benign_label_id()]
    same = torch.allclose(m_manual, out["margin"])
    pred_attack = (lg.argmax(-1) == ad.get_attack_label_id())
    consistent = bool(((out["margin"] > 0) == pred_attack).all())
    print(f"F. margin = z_attack - z_benign 일관성 {same}, "
          f"margin>0 이 attack 예측과 일치 {consistent} -> "
          f"{'통과' if same and consistent else '★실패'}")
    print(f"   margin 범위 {out['margin'].min().item():.3f} ~ {out['margin'].max().item():.3f}"
          f"  (attack 예측 {int(pred_attack.sum())}/{len(pick)}개)")
    return out


if __name__ == "__main__":
    main()

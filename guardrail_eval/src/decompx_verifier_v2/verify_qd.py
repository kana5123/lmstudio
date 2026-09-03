"""STEP 5 + STEP 7: 층 정렬 검증과 sum_k a_lk ~= q_l^T g_l 대수 검사(§24).

이 검사는 g 를 MAIN Query 로 쓰기 위한 것이 아니다.
q-D 구현이 올바른지 확인하는 대수적 sanity check 다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier_v2.base_adapter import PromptGuard2Adapter
from src.decompx_verifier_v2.config import BASE_MODEL, RES, TOL_QD_IDENTITY_REL
from src.decompx_verifier_v2.decompx_features import DecompXTransitionExtractor
from src.decompx_verifier_v2.label_margin_retrieval import LabelMarginRetriever, label_margin_score
from src.decompx_verifier_v2.q_extractor import MarginSensitivityExtractor


def main(n=24, device="cuda"):
    ad = PromptGuard2Adapter(BASE_MODEL, device)
    qx = MarginSensitivityExtractor(ad)
    dx = DecompXTransitionExtractor(ad)
    ret = LabelMarginRetriever()

    d = pd.read_parquet(Path(__file__).resolve().parents[2]
                        / "data/decompx_verifier/pg2_predictions.parquet",
                        columns=["sample_id", "text", "use", "length_ok", "token_length",
                                 "confusion_cell"])
    d = d[(d.use == "MAIN") & d.length_ok & d.token_length.between(20, 220)]
    pick = pd.concat([d[d.confusion_cell == c].head(n // 4) for c in ("TP", "FP", "TN", "FN")])
    enc = ad.encode(pick.text.tolist())
    print(f"표본 {len(pick)}개 (셀 {pick.confusion_cell.value_counts().to_dict()}), "
          f"배치 {tuple(enc['input_ids'].shape)}")

    # DecompX 귀속 텐서가 (B,T,T,d) 라 배치를 작게 끊어 처리한다
    CH = 4
    qs, ds = [], []
    for s0 in range(0, len(pick), CH):
        ii = enc["input_ids"][s0:s0 + CH]
        am = enc["attention_mask"][s0:s0 + CH]
        qs.append(qx.extract(ii, am))
        ds.append(dx.extract(ii, am))
        torch.cuda.empty_cache()
    cat = lambda lst, k: torch.cat([x[k] for x in lst], 0)
    qo = dict(q=cat(qs, "q"), logits=cat(qs, "logits"), margin=cat(qs, "margin"),
              hidden=tuple(torch.cat([x["hidden"][i] for x in qs], 0)
                           for i in range(len(qs[0]["hidden"]))))
    do = dict(D=cat(ds, "D"), H_pre=cat(ds, "H_pre"), C=cat(ds, "C"), logits=cat(ds, "logits"),
              hidden=tuple(torch.cat([x["hidden"][i] for x in ds], 0)
                           for i in range(len(ds[0]["hidden"]))))

    # --- 두 순전파의 은닉상태가 같은가 (q 와 D/H 를 섞어 쓰므로 필수) --------
    hm = max((a - b).abs().max().item() for a, b in zip(qo["hidden"], do["hidden"]))
    lm = (qo["logits"] - do["logits"]).abs().max().item()
    print(f"\n두 순전파 일치: 은닉상태 최대차 {hm:.3e}, 로짓 최대차 {lm:.3e}")

    # --- STEP 5: 층 정렬 ---------------------------------------------------
    K, L = dx.K, ad.get_num_layers()
    print(f"\nSTEP 5 층 정렬:  L={L}, K={K}")
    print(f"  q shape {tuple(qo['q'].shape)}  D shape {tuple(do['D'].shape)}  "
          f"H_pre shape {tuple(do['H_pre'].shape)}")
    ok = True
    for j in [0, 1, K - 1]:
        Dj_ref = (do["C"][:, j + 1] - do["C"][:, j]) * enc["attention_mask"][:, :, None]
        e = (do["D"][:, j] - Dj_ref).abs().max().item()
        Hj_ref = do["hidden"][j + 1]
        eh = (do["H_pre"][:, j] - Hj_ref).abs().max().item()
        ok &= (e == 0.0 and eh == 0.0)
        print(f"  j={j:>2}: D[j] == C[{j+1}]-C[{j}] (차 {e:.1e}) | "
              f"H_pre[j] == hidden[{j+1}] (차 {eh:.1e}) | q[j] = q_L{qx.dest_layers[j]}")
    print(f"  -> {'통과' if ok else '★실패'}")

    # --- STEP 7: sum_k a_lk ~= q_l^T g_l ------------------------------------
    q, D, mask = qo["q"], do["D"], enc["attention_mask"]
    a = label_margin_score(q, D, mask)                       # [B,K,T]
    H = torch.stack(do["hidden"], 1)                          # [B,L+1,T,d]
    g = H[:, 2:L + 1, 0] - H[:, 1:L, 0]                       # [B,K,d]  g_l = h_CLS^l - h_CLS^(l-1)
    qg = torch.einsum("bkd,bkd->bk", q, g)                    # [B,K]
    sa = a.sum(-1)                                            # [B,K]
    abs_err = (sa - qg).abs()
    rel_err = abs_err / (qg.abs() + 1e-12)
    print(f"\nSTEP 7  sum_k a_lk  vs  q_l^T g_l")
    print(f"  절대오차  평균 {abs_err.mean():.3e}  최대 {abs_err.max():.3e}")
    print(f"  상대오차  평균 {rel_err.mean():.3e}  최대 {rel_err.max():.3e}  "
          f"허용 {TOL_QD_IDENTITY_REL:.0e} -> "
          f"{'통과' if rel_err.max() <= TOL_QD_IDENTITY_REL else '★초과'}")
    print(f"  |q^T g| 범위 {qg.abs().min():.3e} ~ {qg.abs().max():.3e}")

    rows = []
    for bi in range(len(pick)):
        for j in range(K):
            rows.append(dict(sample_id=pick.sample_id.iloc[bi],
                             layer_transition=f"L{j+1}->L{j+2}", dest_layer=j + 2,
                             sum_a=float(sa[bi, j]), qTg=float(qg[bi, j]),
                             abs_err=float(abs_err[bi, j]), rel_err=float(rel_err[bi, j])))
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "qd_identity_audit.csv", index=False)
    print(f"  저장 -> {RES/'qd_identity_audit.csv'}")

    # --- 검색 결과 shape ----------------------------------------------------
    ev = ret(q, D, do["H_pre"], mask)
    print("\n검색 결과 shape:")
    for k in ("a", "zD_pos", "zD_neg", "zH_pos", "zH_neg", "mass_pos", "mass_neg"):
        print(f"  {k:<9} {tuple(ev[k].shape)}")
    print(f"  mass_pos 평균 {ev['mass_pos'].mean():.3f}  mass_neg 평균 {ev['mass_neg'].mean():.3f}")
    print(f"  mass 가 0 인 (표본,층) 칸: pos {int((ev['mass_pos']==0).sum())}, "
          f"neg {int((ev['mass_neg']==0).sum())} / {ev['mass_pos'].numel()}")
    return ev


if __name__ == "__main__":
    main()

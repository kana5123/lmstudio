"""층별 DecompX 인코더 분해에서 방향성 토큰 기여 이동(a)을 뽑는다 (지시문 5·7·14·15절).

정의 (이름을 절대 섞지 않는다 — 지시문 18절):
  C_k^(l) : 층 l 최종 인코더 분해(post-LN2)의 CLS 행, 원천토큰 k 의 기여 벡터
  D_k^(l) : C_k^(l) - C_k^(l-1)              Token Contribution Shift
  g^(l)   : h_CLS^(l) - h_CLS^(l-1)          실제 CLS 이동 (일반 은닉표현에서 직접)
  v_U^(l) : ver_train TP-FP 평균차 단위 방향  (여기서 읽어 쓰기만 한다)
  p^(l)   : dot(v, g)                        전역 사영
  a_k^(l) : dot(v_U^(l), D_k^(l))            Directional Token Contribution Shift

주 분석 구간은 전이 L1->L2 … L11->L12 (11개).  임베딩->L1 은 제외.
색인 규약: C[j] = 층 j+1 의 분해.  전이 j (0..10) = L(j+1)->L(j+2) = C[j+1]-C[j].
           이는 v_U 의 색인 j+1 에 대응한다 (v_U[0] 은 임베딩->L1 이라 안 쓴다).

검증(스칼라)도 표본마다 함께 저장한다:
  recon_l   : ||sum_k C_k^(l) - h^(l)|| / ||h^(l)||                     (지시문 4절)
  cons_*    : sum_k D_k^(l) 와 g^(l) 의 절대/상대 오차, 코사인            (지시문 7절)
  projcons_*: sum_k a_k^(l) 와 dot(v,g^(l)) 의 절대/상대 오차             (지시문 15절)
"""
import argparse, json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.build_verifier_dataset import load_model, windows, OUT
from pg2_decompx.decompx_utils import DecompXConfig
from pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/directional_alignment"
MAXN = 512


@torch.no_grad()
def one(wrap, tok, rec, dcfg, V):
    """V: (L, H) — v_U.  전이 j 는 V[j+1] 을 쓴다."""
    enc = windows(tok, rec["text"])
    w = rec["best_window"]
    ids = enc["input_ids"][w:w + 1].to(wrap.model.device)
    msk = enc["attention_mask"][w:w + 1].to(wrap.model.device)
    logits, hidden, hs, out = wrap.forward(ids, msk, dcfg, output_hidden_states=True)
    L = wrap.cfg.num_hidden_layers
    C = out.cls_encoder[0].double()                      # (L, N, H)  C[j]=층 j+1 분해
    H = torch.stack([hs[l][0, 0] for l in range(L + 1)]).double()   # (L+1, H) 0=임베딩

    # --- 지시문 4절: 층별 복원 ---
    recon = ((C.sum(1) - H[1:]).norm(dim=-1) / (H[1:].norm(dim=-1) + 1e-12))   # (L,)

    # --- 전이 j = L(j+1)->L(j+2), j=0..L-2  (11개) ---
    D = C[1:] - C[:-1]                                   # (L-1, N, H)
    g = H[2:] - H[1:L]                                   # (L-1, H)
    v = V[1:].to(C.device).double()                      # (L-1, H)

    sD = D.sum(1)                                        # (L-1, H)
    cons_abs = (sD - g).norm(dim=-1)
    cons_rel = cons_abs / (g.norm(dim=-1) + 1e-12)
    cons_cos = torch.nn.functional.cosine_similarity(sD, g, dim=-1)

    a = torch.einsum("lh,lnh->ln", v, D)                 # (L-1, N)
    p = torch.einsum("lh,lh->l", v, g)                   # (L-1,)
    pj_abs = (a.sum(1) - p).abs()
    pj_rel = pj_abs / (p.abs() + 1e-12)

    return dict(a=a.float().cpu(), mask=msk[0].bool().cpu(), ids=ids[0].cpu(),
                recon=recon.float().cpu(), cons_abs=cons_abs.float().cpu(),
                cons_rel=cons_rel.float().cpu(), cons_cos=cons_cos.float().cpu(),
                p=p.float().cpu(), a_sum=a.sum(1).float().cpu(),
                pj_abs=pj_abs.float().cpu(), pj_rel=pj_rel.float().cpu(),
                g_norm=g.norm(dim=-1).float().cpu(),
                logit_unsafe=float(logits[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a_ = ap.parse_args()

    tok, m = load_model("cuda" if torch.cuda.is_available() else "cpu")
    wrap = DecompXDebertaV2(m)
    vd = torch.load(ART / "v_u.pt", weights_only=False)
    assert vd["fit_split"] == "ver_train", vd["fit_split"]
    V = vd["v"]                                          # (L, H)
    dcfg = DecompXConfig(output_all_layers=True, output_encoder=None,
                         output_classifier=False, ffn_chunk=64)

    rows = [json.loads(l) for l in open(OUT / f"pg2_{a_.split}.jsonl", encoding="utf-8")]
    uns = [r for r in rows if r["base_prediction"] == 1]
    mine = uns[a_.shard::a_.nshards]
    print(f"{a_.split} shard {a_.shard}/{a_.nshards}: {len(mine)}/{len(uns)}건", flush=True)

    N, T = len(mine), V.shape[0] - 1                     # T = 전이 수 (11)
    A = torch.zeros(N, T, MAXN); MK = torch.zeros(N, MAXN, dtype=torch.bool)
    ID = torch.zeros(N, MAXN, dtype=torch.long)
    ck = {k: torch.zeros(N, V.shape[0] if k == "recon" else T)
          for k in ("recon", "cons_abs", "cons_rel", "cons_cos",
                    "p", "a_sum", "pj_abs", "pj_rel", "g_norm")}
    t0 = time.time()
    for i, r in enumerate(mine):
        o = one(wrap, tok, r, dcfg, V)
        n = o["a"].shape[1]
        A[i, :, :n] = o["a"]; MK[i, :n] = o["mask"]; ID[i, :n] = o["ids"]
        for k in ck:
            ck[k][i] = o[k]
        assert abs(o["logit_unsafe"] - r["logit_unsafe"]) < 1e-3, r["sample_id"]
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N} {el/(i+1):.2f}s/건 남은 {(N-i-1)*el/(i+1)/60:.1f}분 "
                  f"복원max={ck['recon'][:i+1].max():.2e} "
                  f"보존rel_max={ck['cons_rel'][:i+1].max():.2e}", flush=True)
    d = {"sample_id": [r["sample_id"] for r in mine],
         "gt": torch.tensor([r["gt"] for r in mine]),
         "a": A, "mask": MK, "input_ids": ID, "n_transitions": T, **ck}
    p = ART / f"dir_{a_.split}_{a_.shard}of{a_.nshards}.pt"
    torch.save(d, p)
    print(f"저장 {p}  복원max={ck['recon'].max():.3e}  보존rel max={ck['cons_rel'].max():.3e} "
          f"cos min={ck['cons_cos'].min():.6f}  사영보존 rel max={ck['pj_rel'].max():.3e}")


if __name__ == "__main__":
    main()

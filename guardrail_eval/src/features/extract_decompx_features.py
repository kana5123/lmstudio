"""DecompX 로 토큰별 CLS 이동 기여를 뽑는다.

각 UNSAFE 표본에 대해 (PG2 가 최고점을 낸 창 하나에서):
    C_cls^(1)[k]  = 1번째 층 CLS 표현에 대한 입력토큰 k 의 기여 벡터
    C_cls^(L)[k]  = 마지막 층에 대한 같은 것
    delta_c_k     = C_cls^(L)[k] - C_cls^(1)[k]                    (768차원)
    directional_score_k    = dot(delta_c_k, v)                     (스칼라)
    unsafe_margin_contrib_k= 분류기기여[k,UNSAFE] - 분류기기여[k,BENIGN]  (스칼라)

**검증**: sum_k delta_c_k == h_cls^(L) - h_cls^(1) 인지 표본마다 확인해 최대오차를 보고한다.
편향은 bias_decomposer 가 토큰들에게 나눠 넣으므로 **별도 편향 항이 없다**.

방향 v 는 ver_train 에서만 만든 것을 읽어 쓴다(테스트 누수 금지).

병렬 실행:  --shard i --nshards n  으로 쪼개고 나중에 합친다.
"""
import argparse, json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.build_verifier_dataset import load_model, windows, OUT, UNSAFE_ID, BENIGN_ID
from pg2_decompx.decompx_utils import DecompXConfig
from pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

DIRS = Path(__file__).resolve().parents[2] / "artifacts" / "directions"
MAXN = 512


@torch.no_grad()
def one(wrap, tok, rec, dcfg, v):
    enc = windows(tok, rec["text"])
    w = rec["best_window"]
    ids = enc["input_ids"][w:w + 1].to(wrap.model.device)
    msk = enc["attention_mask"][w:w + 1].to(wrap.model.device)
    logits, hidden, hs, out = wrap.forward(ids, msk, dcfg, output_hidden_states=True)
    L = wrap.cfg.num_hidden_layers
    c1, cL = out.cls_encoder[0, 0], out.cls_encoder[0, L - 1]     # (N,768) 각각 1층/L층
    delta_c = cL - c1                                              # (N,768)
    # 복원 검증: 토큰 기여 합 == 실제 CLS 이동
    ref = hs[L][0, 0] - hs[1][0, 0]
    err = (delta_c.sum(0) - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    margin = out.classifier[0, :, UNSAFE_ID] - out.classifier[0, :, BENIGN_ID]   # (N,)
    return dict(delta_c=delta_c.half().cpu(),
                directional=(delta_c @ v).float().cpu(),
                margin=margin.float().cpu(),
                mask=msk[0].cpu(),
                ids=ids[0].cpu(),
                n=int(msk.sum()), rel_err=err,
                logit_unsafe=float(logits[0, UNSAFE_ID]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ffn_chunk", type=int, default=64)
    a = ap.parse_args()

    tok, m = load_model("cuda" if torch.cuda.is_available() else "cpu")
    wrap = DecompXDebertaV2(m)
    v = torch.load(DIRS / "pg2_tp_fp_delta_direction.pt", weights_only=False)["v"].to(m.device)
    dcfg = DecompXConfig(output_all_layers=True, output_encoder=None,
                         output_classifier=True, ffn_chunk=a.ffn_chunk)

    rows = [json.loads(l) for l in open(OUT / f"pg2_{a.split}.jsonl", encoding="utf-8")]
    uns = [r for r in rows if r["base_prediction"] == 1]
    mine = uns[a.shard::a.nshards]
    print(f"{a.split} shard {a.shard}/{a.nshards}: {len(mine)}/{len(uns)}건", flush=True)

    N = len(mine)
    dc = torch.zeros(N, MAXN, 768, dtype=torch.float16)
    ds = torch.zeros(N, MAXN); mg = torch.zeros(N, MAXN)
    mk = torch.zeros(N, MAXN, dtype=torch.bool); tid = torch.zeros(N, MAXN, dtype=torch.long)
    lens, errs = [], []
    t0 = time.time()
    for i, r in enumerate(mine):
        o = one(wrap, tok, r, dcfg, v)
        n = o["delta_c"].shape[0]
        dc[i, :n] = o["delta_c"]; ds[i, :n] = o["directional"]
        mg[i, :n] = o["margin"];  mk[i, :n] = o["mask"].bool(); tid[i, :n] = o["ids"]
        # 주의: n 은 **창 폭(패딩 포함)** 이다.  실제 토큰 수는 mask 로 세야 한다.
        # (기존 저장 파일도 같은 규약이므로 소비 측에서 mask.sum() 을 쓴다)
        lens.append(n); errs.append(o["rel_err"])
        assert abs(o["logit_unsafe"] - r["logit_unsafe"]) < 1e-3, (r["sample_id"], o, r)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N}  {el/(i+1):.2f}s/건  남은 {(N-i-1)*el/(i+1)/60:.1f}분  "
                  f"복원상대오차 max={max(errs):.2e}", flush=True)
    d = {"sample_id": [r["sample_id"] for r in mine],
         "gt": torch.tensor([r["gt"] for r in mine]),
         "unsafe_probability": torch.tensor([r["unsafe_probability"] for r in mine]),
         "logit_margin": torch.tensor([r["logit_margin"] for r in mine]),
         "delta_c": dc, "directional": ds, "margin": mg, "mask": mk, "input_ids": tid,
         "seq_len": torch.tensor(lens), "recon_rel_err": torch.tensor(errs)}
    p = OUT / f"decompx_{a.split}_{a.shard}of{a.nshards}.pt"
    torch.save(d, p)
    print(f"저장 {p}  복원 상대오차 max={max(errs):.3e} mean={sum(errs)/len(errs):.3e}")


if __name__ == "__main__":
    main()

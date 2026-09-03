"""검증기 학습 + 12장 절제 실험.

규약:
  - 모든 스케일러/방향/임계값은 **ver_train** 에서만 적합한다.
  - 모델 선택(조기 종료)은 **ver_dev** 로 한다.  eval_val 은 최종 임계값 선택 전용,
    eval_test 는 최종 보고 전용이며 학습·선택에 일절 쓰지 않는다.
  - 씨앗 3개를 돌려 평균±표준편차를 보고한다(단일 체크포인트 비교는 잡음에 뒤집힌다).

절제 구성(지시문 12장):
  B0 PG2 점수만 / B1 hL만 / B2 delta_h만 / B3 전역delta+점수 /
  B4 DecompX 분류기기여 요약만 / B5 DecompX 방향성 토큰신호만 /
  B6 전역delta+DecompX토큰 / B7 제안 전체 / B7ctrl 용량대조(토큰특징 무작위 섞기)
"""
import argparse, copy, json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from analysis.metrics import auroc, auprc
from models.global_delta_verifier import GlobalDeltaVerifier
from models.token_decomp_verifier import TokenDecompVerifier
from train.dataset import load_split, Scaler, VerifierDS

ROOT = Path(__file__).resolve().parents[2]
RES, MOD = ROOT / "artifacts/results", ROOT / "artifacts/models"
SEEDS = (0, 1, 2)

# name -> (global_feat, kwargs, need_tokens)
ABLATIONS = {
    "B1_hL_only":        ("hL",      dict(use_global=True,  use_token=False, use_numeric=False), False),
    "B2_delta_only":     ("delta_h", dict(use_global=True,  use_token=False, use_numeric=False), False),
    "B3_delta_score":    ("delta_h", dict(use_global=True,  use_token=False, use_numeric=True),  False),
    "B4_decompx_margin": ("delta_h", dict(use_global=False, use_token=True,  use_numeric=False,
                                          token_vec=False, token_dir=False, token_margin=True), True),
    "B5_decompx_dir":    ("delta_h", dict(use_global=False, use_token=True,  use_numeric=False,
                                          token_vec=False, token_dir=True,  token_margin=False), True),
    "B5b_decompx_token_only": ("delta_h", dict(use_global=False, use_token=True, use_numeric=False), True),
    "B6_delta_token":    ("delta_h", dict(use_global=True,  use_token=True,  use_numeric=False), True),
    "B7_full":           ("delta_h", dict(use_global=True,  use_token=True,  use_numeric=True),  True),
    "B7ctrl_shuffled":   ("delta_h", dict(use_global=True,  use_token=True,  use_numeric=True),  True),
    # 주의: A_global_delta 는 B3_delta_score 와 **같은 신경망**이다.
    # 지시문 10장의 Verifier A 사양(LayerNorm->128->GELU / 2->16->GELU / 144->64->1)이
    # TokenDecompVerifier(use_global,use_numeric) 와 층 구성·파라미터 수(109,361)까지
    # 일치한다.  독립적인 두 결과가 아니라 **같은 모델의 두 이름**이므로 그렇게 보고한다.
    "A_global_delta":    ("delta_h", None, False),
}


def collate(b):
    o = {}
    for k in b[0]:
        o[k] = torch.stack([x[k] for x in b])
    return o


def shuffle_tokens(d, seed):
    """용량 대조군: 토큰 특징만 표본 사이에서 무작위로 섞어 신호를 죽인다.
    파라미터 수와 계산량은 그대로이므로, 성능이 유지되면 '용량 덕'이라는 뜻이다."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(d["y"]), generator=g)
    d = dict(d)
    for k in ("delta_c", "directional", "margin", "mask"):
        d[k] = d[k][p]
    return d


@torch.no_grad()
def predict(model, ds, dev, bs=64):
    model.eval()
    out, ys = [], []
    for b in DataLoader(ds, batch_size=bs, collate_fn=collate):
        y = b.pop("y")
        out.append(torch.sigmoid(model({k: v.to(dev) for k, v in b.items()})).cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def train_one(name, seed, data, args, dev):
    gfeat, kw, need_tok = ABLATIONS[name]
    torch.manual_seed(seed); np.random.seed(seed)
    tr, dv = data[gfeat]["ver_train"], data[gfeat]["ver_dev"]
    va, te = data[gfeat]["eval_val"], data[gfeat]["eval_test"]
    if name == "B7ctrl_shuffled":
        tr = shuffle_tokens(tr, seed); dv = shuffle_tokens(dv, seed + 100)
        va = shuffle_tokens(va, seed + 200); te = shuffle_tokens(te, seed + 300)

    sc = Scaler().fit(tr, need_tok)
    dss = {n: VerifierDS(x, sc, need_tok) for n, x in
           (("ver_train", tr), ("ver_dev", dv), ("eval_val", va), ("eval_test", te))}

    H = tr["global"].shape[1]
    model = (GlobalDeltaVerifier(H) if kw is None else TokenDecompVerifier(H, **kw)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())

    y = tr["y"].numpy()
    pw = torch.tensor([(1 - y).sum() / max(y.sum(), 1)], device=dev) if args.pos_weight else None
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dl = DataLoader(dss["ver_train"], batch_size=args.bs, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(seed))

    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        for b in dl:
            yb = b.pop("y").to(dev)
            opt.zero_grad()
            lossf(model({k: v.to(dev) for k, v in b.items()}), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        s, yy = predict(model, dss["ver_dev"], dev)
        a = auroc(yy, s)
        if a > best:
            best, best_state, bad = a, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    out = {"name": name, "seed": seed, "params": nparam, "dev_auroc": best, "epochs": ep + 1}
    scores = {}
    for split in ("ver_dev", "eval_val", "eval_test"):
        s, yy = predict(model, dss[split], dev)
        scores[split] = (s, yy, dss[split].d["sample_id"])
        out[f"{split}_auroc"] = auroc(yy, s); out[f"{split}_auprc"] = auprc(yy, s)
    return out, scores, model, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--pos_weight", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    names = [a.only] if a.only else list(ABLATIONS)
    need_g = {ABLATIONS[n][0] for n in names}
    need_tok = any(ABLATIONS[n][2] for n in names)
    data = {}
    for g in need_g:
        data[g] = {s: load_split(s, g, need_tok) for s in
                   ("ver_train", "ver_dev", "eval_val", "eval_test")}
        print(f"적재 완료 global={g}  " + "  ".join(
            f"{s}:{len(data[g][s]['y'])}" for s in data[g]))

    # B0: 학습 없는 기준선 — PG2 점수 그대로
    g0 = data[list(need_g)[0]]
    rows = []
    for split in ("ver_dev", "eval_val", "eval_test"):
        d = g0[split]
        s, yy = d["numeric"][:, 0].numpy(), d["y"].numpy()
        rows.append({"name": "B0_pg2_score", "seed": -1, "params": 0, "split": split,
                     "auroc": auroc(yy, s), "auprc": auprc(yy, s)})
    print(f"B0 PG2 점수만: " + "  ".join(f"{r['split']} AUROC={r['auroc']:.4f}" for r in rows))

    RES.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
    allres, allscores = [], {}
    for name in names:
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            r, sc_out, model, scaler = train_one(name, seed, data, a, dev)
            r["sec"] = round(time.time() - t0, 1)
            runs.append(r); allres.append(r)
            if seed == SEEDS[0]:
                torch.save({"state_dict": model.state_dict(), "scaler": scaler.state(),
                            "ablation": name, "cfg": ABLATIONS[name]}, MOD / f"{name}_seed{seed}.pt")
            allscores[f"{name}_s{seed}"] = {sp: {"score": v[0].tolist(), "y": v[1].tolist(),
                                                 "sample_id": v[2]} for sp, v in sc_out.items()}
        m = lambda k: (float(np.mean([x[k] for x in runs])), float(np.std([x[k] for x in runs])))
        te_m, te_s = m("eval_test_auroc"); tp_m, tp_s = m("eval_test_auprc")
        dv_m, _ = m("ver_dev_auroc")
        print(f"{name:20} params={runs[0]['params']:>8,}  ver_dev AUROC={dv_m:.4f}  "
              f"eval_test AUROC={te_m:.4f}±{te_s:.4f}  AUPRC={tp_m:.4f}±{tp_s:.4f}", flush=True)

    # 다른 스크립트(B8/B9)가 같은 파일에 쓰므로 **덮어쓰지 말고 합친다**
    prev_runs = json.loads((RES / "ablation_runs.json").read_text())["runs"] \
        if (RES / "ablation_runs.json").exists() else []
    keep = [r for r in prev_runs if r["name"] not in names]
    (RES / "ablation_runs.json").write_text(json.dumps({"b0": rows, "runs": keep + allres},
                                                       ensure_ascii=False, indent=1))
    prev = torch.load(RES / "verifier_scores.pt", weights_only=False) \
        if (RES / "verifier_scores.pt").exists() else {}
    prev.update(allscores)
    torch.save(prev, RES / "verifier_scores.pt")
    print(f"저장 -> {RES/'ablation_runs.json'}, {RES/'verifier_scores.pt'}")


if __name__ == "__main__":
    main()

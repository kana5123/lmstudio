"""DecompX 를 오프라인 교사로 쓰는 증류 (지시문 15장).

배포 시 DecompX 를 돌리지 않고, PG2 가 이미 계산한 토큰별 은닉표현에서
DecompX 신호를 **예측**하는 작은 머리를 학습한다.

학생 입력 : [h_k^(1) ; h_k^(L)]  (1536)
학생 출력 : delta_c_k (768) , directional_k (1) , margin_k (1)
학습 대상 : ver_train 의 진짜 DecompX 신호 (교사)

그 다음 **같은 검증기 B 구조**에 진짜 신호 대신 예측 신호를 넣어 학습/평가해
  (a) 완전 DecompX 검증기  vs  (b) 증류 경량 검증기
의 성능과 지연시간을 모두 비교한다.
"""
import argparse, json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn
from analysis.metrics import auroc, auprc
from train.dataset import load_split, Scaler, VerifierDS
from train.train_verifier import collate, predict
from models.token_decomp_verifier import TokenDecompVerifier

ROOT = Path(__file__).resolve().parents[2]
FEAT, RES, MOD = ROOT / "artifacts/features", ROOT / "artifacts/results", ROOT / "artifacts/models"
SPLITS = ("ver_train", "ver_dev", "eval_val", "eval_test")


class TokenHead(nn.Module):
    """h^(1),h^(L) -> DecompX 신호 예측.  토큰마다 독립이라 계산이 싸다."""

    def __init__(self, d_in=1536, hid=256, d_out=768):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hid), nn.GELU(),
                                  nn.Linear(hid, hid), nn.GELU())
        self.vec = nn.Linear(hid, d_out)
        self.sca = nn.Linear(hid, 2)

    def forward(self, x):
        z = self.body(x)
        s = self.sca(z)
        return self.vec(z), s[..., 0], s[..., 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ver_epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = {s: load_split(s, "delta_h", True) for s in SPLITS}
    th = {}
    for s in SPLITS:
        d = torch.load(FEAT / f"tokenhidden_{s}.pt", weights_only=False)
        idx = {k: i for i, k in enumerate(d["sample_id"])}
        o = [idx[k] for k in data[s]["sample_id"]]        # DecompX 순서에 맞춤
        th[s] = torch.cat([d["h1"][o], d["hL"][o]], -1)   # (n, N, 1536)
        print(f"{s}: 토큰은닉 {tuple(th[s].shape)}")

    sc = Scaler().fit(data["ver_train"], True)            # ver_train 에서만 적합
    out = {}

    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        head = TokenHead().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
        tr = data["ver_train"]; n = len(tr["y"])
        t0 = time.time()
        for ep in range(a.epochs):
            head.train(); perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, a.bs):
                b = perm[i:i + a.bs]
                x = th["ver_train"][b].float().to(dev)
                m = tr["mask"][b].to(dev)
                ty = (tr["delta_c"][b].float().to(dev) / sc.c_sd)
                td = ((tr["directional"][b].to(dev) - sc.d_mu) / sc.d_sd)
                tm = ((tr["margin"][b].to(dev) - sc.m_mu) / sc.m_sd)
                pv, pd, pm = head(x)
                mm = m.unsqueeze(-1).float()
                loss = (((pv - ty) ** 2 * mm).sum() / mm.sum().clamp_min(1) / 768
                        + ((pd - td) ** 2 * m).sum() / m.sum().clamp_min(1)
                        + ((pm - tm) ** 2 * m).sum() / m.sum().clamp_min(1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
                tot += float(loss) * len(b)
            print(f"  증류 seed{seed} ep{ep+1} loss={tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)

        # --- 예측 신호로 데이터 교체 ---
        head.eval()
        pred = {}
        with torch.no_grad():
            for s in SPLITS:
                V, D, M = [], [], []
                for i in range(0, len(data[s]["y"]), 16):
                    x = th[s][i:i + 16].float().to(dev)
                    v, dd, mm2 = head(x)
                    V.append((v * sc.c_sd).half().cpu())
                    D.append((dd * sc.d_sd + sc.d_mu).cpu()); M.append((mm2 * sc.m_sd + sc.m_mu).cpu())
                pred[s] = dict(data[s])
                pred[s]["delta_c"] = torch.cat(V)
                pred[s]["directional"] = torch.cat(D)
                pred[s]["margin"] = torch.cat(M)
        # 교사 대비 상관계수 (신호를 얼마나 재현했나)
        cors = {}
        for s in ("ver_dev", "eval_test"):
            m = data[s]["mask"]
            for k in ("directional", "margin"):
                x = pred[s][k][m].numpy(); y = data[s][k][m].numpy()
                cors[f"{s}_{k}_r"] = float(np.corrcoef(x, y)[0, 1])
            x = pred[s]["delta_c"][m].float().numpy().ravel()
            y = data[s]["delta_c"][m].float().numpy().ravel()
            cors[f"{s}_delta_c_r"] = float(np.corrcoef(x, y)[0, 1])
        print(f"  교사-학생 상관: " + "  ".join(f"{k}={v:.3f}" for k, v in cors.items()), flush=True)

        # --- 예측 신호로 검증기 B 학습 ---
        sc2 = Scaler().fit(pred["ver_train"], True)
        dss = {s: VerifierDS(pred[s], sc2, True) for s in SPLITS}
        model = TokenDecompVerifier(768).to(dev)
        opt2 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        lf = nn.BCEWithLogitsLoss()
        from torch.utils.data import DataLoader
        dl = DataLoader(dss["ver_train"], batch_size=32, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
        best, bstate, bad = -1, None, 0
        for ep in range(a.ver_epochs):
            model.train()
            for b in dl:
                y = b.pop("y").to(dev)
                opt2.zero_grad(); lf(model({k: v.to(dev) for k, v in b.items()}), y).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt2.step()
            s_, y_ = predict(model, dss["ver_dev"], dev)
            av = auroc(y_, s_)
            if av > best: best, bstate, bad = av, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8: break
        model.load_state_dict(bstate)
        r = {"seed": seed, "ver_dev_auroc": best, "corr": cors,
             "head_params": sum(p.numel() for p in head.parameters()),
             "verifier_params": sum(p.numel() for p in model.parameters())}
        scores = {}
        for s in ("ver_dev", "eval_val", "eval_test"):
            sc_, y_ = predict(model, dss[s], dev)
            r[f"{s}_auroc"] = auroc(y_, sc_); r[f"{s}_auprc"] = auprc(y_, sc_)
            scores[s] = {"score": sc_.tolist(), "y": y_.tolist(), "sample_id": pred[s]["sample_id"]}
        out[f"seed{seed}"] = r
        print(f"증류 검증기 seed{seed}: eval_test AUROC={r['eval_test_auroc']:.4f}", flush=True)
        prev = torch.load(RES / "verifier_scores.pt", weights_only=False)
        prev[f"B9_distilled_s{seed}"] = scores
        torch.save(prev, RES / "verifier_scores.pt")
        torch.save({"head": head.state_dict(), "verifier": model.state_dict()},
                   MOD / f"distilled_seed{seed}.pt")

    te = [v["eval_test_auroc"] for v in out.values()]
    print(f"\n증류 검증기 eval_test AUROC={np.mean(te):.4f}±{np.std(te):.4f}")
    (RES / "distill.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

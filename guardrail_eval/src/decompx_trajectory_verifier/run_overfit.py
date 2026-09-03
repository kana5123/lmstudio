"""§23 STEP 7 / §24 small overfit test.

목적은 성능 측정이 아니라 label routing / mask / tensor permutation / forward /
loss / optimizer 버그를 찾는 것이다.  같은 subset 으로 학습하고 평가한다.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.train import (CacheDataset, build_model, collate,
                                                   evaluate, make_perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ART / "pilot_256.pt"))
    ap.add_argument("--variant", default="A3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    ds = CacheDataset([Path(a.cache)])
    print(f"표본 {len(ds)}  FP {int(ds.y.sum())}  TP {int((ds.y==0).sum())}  "
          f"L={ds.L} d={ds.d}  출처 {sorted(set(ds.source))}")
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate)
    dl_eval = DataLoader(ds, batch_size=a.bs, shuffle=False, collate_fn=collate)

    model = build_model(ds, a.variant, dev=a.dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"변형 {a.variant}  파라미터 {npar:,}  (드롭아웃 유지, 같은 데이터로 학습·평가)\n")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    lossf = nn.BCEWithLogitsLoss()
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot, n = 0.0, 0
        for Cx, fa, mk, y in dl:
            Cx, fa, mk, y = Cx.to(a.dev), fa.to(a.dev), mk.to(a.dev), y.to(a.dev)
            perm = make_perm(mk.cpu()).to(a.dev) if a.variant == "A5" else None
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Cx, fa, mk, perm), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tot += float(loss) * len(y); n += len(y)
        if ep % 5 == 0 or ep == a.epochs - 1:
            m, _, _ = evaluate(model, dl_eval, a.dev, a.variant)
            hist.append(dict(epoch=ep, train_loss=tot / n, **m))
            print(f"  ep{ep:>2}  train {tot/n:.4f}  eval loss {m['loss']:.4f}  "
                  f"AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}")
    m, S, Y = evaluate(model, dl_eval, a.dev, a.variant)
    print(f"\n최종  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  손실 {m['loss']:.5f}  "
          f"({time.time()-t0:.0f}s)")
    acc = float(((S >= 0.5).astype(int) == Y).mean())
    print(f"      정확도 {acc:.4f}  (n={m['n']}, FP {m['n_fp']})")
    import pandas as pd
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(RES / f"small_overfit_{a.variant}.csv", index=False)
    return m


if __name__ == "__main__":
    main()

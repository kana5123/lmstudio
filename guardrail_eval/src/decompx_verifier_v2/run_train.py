"""STEP 15: 전체 verifier 학습."""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_verifier_v2.config import ART, RES
from src.decompx_verifier_v2.train import EvidenceDataset, run

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--wd", type=float, default=1e-2)
a = ap.parse_args()

paths = sorted((ART / "cache").glob("ev_s*_c*.pt"))
print(f"캐시 샤드 {len(paths)}개")
for sp in ("train", "val", "test"):
    ds = EvidenceDataset(paths, sp)
    print(f"  {sp:<6} {len(ds):>7,}  attack분기 {int((ds.base_pred==1).sum()):>6,}  "
          f"benign분기 {int((ds.base_pred==0).sum()):>6,}  "
          f"셀 { {c: ds.cell.count(c) for c in ('TP','FP','TN','FN')} }")
print()
m, r = run(paths, paths, paths, epochs=a.epochs, patience=5, bs=a.bs, lr=a.lr, wd=a.wd,
           seed=a.seed, splits=("train", "val", "test"),
           out_json=RES / f"train_seed{a.seed}.json")
for nm in ("val", "test"):
    v = r[nm]
    print(f"\n[{nm}]  손실 {v['loss']:.4f}")
    print(f"  attack 분기 (TP vs FP)  AUROC {v['attack_auroc']:.4f}  AUPRC {v['attack_auprc']:.4f}"
          f"  n={v['attack_n']:,}  양성(FP)={v['attack_pos']:,}")
    print(f"  benign 분기 (TN vs FN)  AUROC {v['benign_auroc']:.4f}  AUPRC {v['benign_auprc']:.4f}"
          f"  n={v['benign_n']:,}  양성(FN)={v['benign_pos']:,}")
import torch
torch.save(m.state_dict(), ART / f"verifier_seed{a.seed}.pt")
print(f"\n모델 저장 -> {ART/f'verifier_seed{a.seed}.pt'}")

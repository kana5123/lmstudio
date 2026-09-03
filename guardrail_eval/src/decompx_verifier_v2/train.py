"""Verifier 학습(§29).  base 는 캐시된 evidence 로만 등장하므로 학습 대상이 아니다."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_verifier_v2 import config as C
from src.decompx_verifier_v2.depth_verifier import DepthVerifier
from src.decompx_verifier_v2.losses import CELL_TARGET, branch_loss, branch_scores

KEYS = ("q", "zD_pos", "zD_neg", "zH_pos", "zH_neg", "mass_pos", "mass_neg")


class EvidenceDataset(Dataset):
    def __init__(self, paths, split=None):
        """split 을 주면 그 분할의 행만 남긴다.  캐시 파일에 split 이 들어 있다."""
        ev = {k: [] for k in KEYS}
        cell, bp, dsn, sid = [], [], [], []
        K = d = None
        for p in paths:
            b = torch.load(p, weights_only=False)
            K, d = b["K"], b["d"]
            if split is not None and b.get("split") is not None:
                m = torch.tensor([s == split for s in b["split"]])
                if not m.any():
                    continue
            else:
                m = torch.ones(len(b["confusion_cell"]), dtype=torch.bool)
            for k in KEYS:
                ev[k].append(b[k][m].float())
            idx = m.nonzero(as_tuple=True)[0].tolist()
            cell += [b["confusion_cell"][i] for i in idx]
            dsn += [b["dataset"][i] for i in idx]
            sid += [b["sample_id"][i] for i in idx]
            bp.append(b["base_pred"][m])
        assert cell, f"split={split} 에 해당하는 행이 없다"
        self.ev = {k: torch.cat(v, 0) for k, v in ev.items()}
        self.cell, self.dataset, self.sample_id = cell, dsn, sid
        self.base_pred = torch.cat(bp).long()
        self.target = torch.tensor([CELL_TARGET[c] for c in cell], dtype=torch.float32)
        self.K, self.d = K, d

    def __len__(self):
        return len(self.cell)

    def __getitem__(self, i):
        x = {k: self.ev[k][i] for k in KEYS}
        return x, self.base_pred[i], self.target[i]


def collate(batch):
    xs, bp, tg = zip(*batch)
    return ({k: torch.stack([x[k] for x in xs]) for k in KEYS},
            torch.stack(bp), torch.stack(tg))


def metrics(scores, base_pred, target, cells):
    """분기별 AUROC/AUPRC.  양성 = base 예측이 틀림."""
    out = {}
    for name, m in (("attack", base_pred == 1), ("benign", base_pred == 0)):
        y, s = target[m].numpy(), scores[m].numpy()
        if len(np.unique(y)) < 2:
            out[f"{name}_auroc"] = out[f"{name}_auprc"] = float("nan")
        else:
            out[f"{name}_auroc"] = roc_auc_score(y, s)
            out[f"{name}_auprc"] = average_precision_score(y, s)
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_pos"] = int(y.sum())
    return out


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    S, P, T, tot, n = [], [], [], 0.0, 0
    for x, bp, tg in loader:
        x = {k: v.to(dev) for k, v in x.items()}
        bp, tg = bp.to(dev), tg.to(dev)
        o = model(x)
        tot += float(branch_loss(o, bp, tg)) * len(tg); n += len(tg)
        S.append(branch_scores(o, bp).cpu()); P.append(bp.cpu()); T.append(tg.cpu())
    S, P, T = torch.cat(S), torch.cat(P), torch.cat(T)
    return tot / n, metrics(S, P, T, None), S


def run(train_paths, val_paths, test_paths=None, epochs=30, patience=5, bs=256, lr=1e-4,
        wd=1e-2, dev="cuda", seed=0, dropout=None, log=print, out_json=None,
        splits=(None, None, None)):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = EvidenceDataset(train_paths, splits[0])
    va = EvidenceDataset(val_paths, splits[1])
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate, drop_last=False)
    dl_va = DataLoader(va, batch_size=512, shuffle=False, collate_fn=collate)
    model = DepthVerifier(tr.d, tr.K, C.PROJ_DIM, C.FUSION_HIDDEN, C.D_MODEL, C.NHEAD,
                          C.NUM_LAYERS, C.DIM_FEEDFORWARD,
                          C.DROPOUT if dropout is None else dropout,
                          C.NORM_FIRST, C.DEPTH_POS_ENCODING).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(epochs):
        model.train(); tot, n = 0.0, 0
        for x, bp, tg in dl_tr:
            x = {k: v.to(dev) for k, v in x.items()}
            bp, tg = bp.to(dev), tg.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = branch_loss(model(x), bp, tg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(tg); n += len(tg)
        vl, vm, _ = evaluate(model, dl_va, dev)
        # 분기 균형 val AUROC 로 조기 종료 판정
        sel = np.nanmean([vm["attack_auroc"], vm["benign_auroc"]])
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl, **vm, sel=sel))
        log(f"  ep{ep:>2} train {tot/n:.4f}  val {vl:.4f}  "
            f"attack AUROC {vm['attack_auroc']:.4f}  benign AUROC {vm['benign_auroc']:.4f}  "
            f"평균 {sel:.4f}")
        if sel > best:
            best, bad = sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"  조기 종료 (patience {patience})"); break
    model.load_state_dict(best_state)
    res = dict(best_val_sel=best, history=hist)
    for nm, paths, sp in (("val", val_paths, splits[1]), ("test", test_paths, splits[2])):
        if not paths:
            continue
        ds = EvidenceDataset(paths, sp)
        l, m, _ = evaluate(model, DataLoader(ds, batch_size=512, collate_fn=collate), dev)
        res[nm] = dict(loss=l, **m)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    return model, res

"""PHASE C1 학습: M0 / A0 / A3 (§10-§13).  아키텍처는 동결된 것을 그대로 쓴다."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.config import ART
from src.decompx_trajectory_verifier.data import (A_TOL, BalancedBucketSampler, EvidenceDataset,
                                                  LengthBucketSampler, MemmapEvidence, collate)
from src.decompx_trajectory_verifier.margin_model import MarginOnly
from src.decompx_trajectory_verifier.metrics import core_metrics
from src.decompx_trajectory_verifier.model import DXTV

NEEDS_C = {"M0": False, "A0": False, "A3": True,
           "V0": False, "V1": True, "V2": True, "V3": True, "V4": True}


def build(model_name, ds, dev):
    if model_name == "M0":
        m = MarginOnly().to(dev)
        z = ds.logits
        f = np.stack([z[:, 0], z[:, 1], z[:, 1] - z[:, 0]], -1)
        m.set_stats(f.mean(0), f.std(0))
        return m
    m = DXTV(ds.mm.d, ds.mm.L, 512, variant=model_name, d_v=C.D_V, depth_tf=C.DEPTH_TF,
             token_tf=C.TOKEN_TF, attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT,
             head_hidden=C.HEAD_HIDDEN).to(dev)
    if m.cfgv["use_attr"]:                       # §9 정규화 통계는 train fold 에서만
        acc = []
        for i in range(len(ds)):
            acc.append(ds[i][1])
        x = torch.cat(acc, 0)
        m.anchor.set_stats(x.mean(0), x.std(0))
    return m


def fwd(model, name, C_, fa, mk, lg):
    return model(lg) if name == "M0" else model(C_, fa, mk, None)


@torch.no_grad()
def predict(model, name, ds, dev, bs=32):
    model.eval()
    sam = LengthBucketSampler(ds, bs)
    order, P = [], []
    dl = DataLoader(ds, batch_sampler=sam, collate_fn=collate, num_workers=4)
    for b, batch in zip(sam, dl):
        C_, fa, mk, lg, y, _ = batch
        p = torch.sigmoid(fwd(model, name, C_.to(dev), fa.to(dev), mk.to(dev), lg.to(dev)))
        order += list(b); P.append(p.float().cpu())
    P = torch.cat(P).numpy()
    out = np.zeros(len(ds), dtype=np.float32)
    out[np.array(order)] = P
    return out


def macro_auprc(ds, p):
    vals = []
    for s in np.unique(ds.source):
        m = ds.source == s
        r = core_metrics(ds.y[m], p[m])
        if not np.isnan(r["auprc"]):
            vals.append(r["auprc"])
    return float(np.mean(vals)) if vals else float("nan")


def run(model_name, manifest, seed, dev="cuda", bs=32, log=print, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    mm = MemmapEvidence()
    need = NEEDS_C[model_name]
    tr = EvidenceDataset(mm, manifest, "train", need_C=need)
    va = EvidenceDataset(mm, manifest, "val", need_C=need)
    te = EvidenceDataset(mm, manifest, "test", need_C=need)
    model = build(model_name, tr, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    sam = BalancedBucketSampler(tr, bs, seed=seed)
    dl = DataLoader(tr, batch_sampler=sam, collate_fn=collate, num_workers=6,
                    persistent_workers=True)
    best, best_state, bad, hist, a_dev = -np.inf, None, 0, [], 0.0
    for ep in range(C.MAX_EPOCHS):
        model.train()
        tot = n = ntok = 0
        t_ep = time.time(); wait = 0.0; t_last = time.time()
        for C_, fa, mk, lg, y, adev in dl:
            wait += time.time() - t_last
            a_dev = max(a_dev, adev)
            C_, fa, mk, lg, y = (C_.to(dev), fa.to(dev), mk.to(dev), lg.to(dev), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss = lossf(fwd(model, model_name, C_, fa, mk, lg), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tot += float(loss) * len(y); n += len(y); ntok += int(mk.sum())
            t_last = time.time()
        el = time.time() - t_ep
        assert a_dev <= A_TOL, f"a_runtime 과 a_cached 가 {a_dev:.2e} 만큼 다르다 (허용 {A_TOL:.0e})"
        pv = predict(model, model_name, va, dev, bs)
        vl = float(nn.functional.binary_cross_entropy(
            torch.tensor(pv).clamp(1e-7, 1 - 1e-7), torch.tensor(va.y)))
        mac = macro_auprc(va, pv)
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl, val_macro_auprc=mac,
                         examples_per_sec=n / el, tokens_per_sec=ntok / el,
                         data_wait_ratio=wait / el, epoch_sec=el))
        log(f"    ep{ep:>2} train {tot/n:.4f} val {vl:.4f} macroAUPRC {mac:.4f} "
            f"({n/el:.0f} ex/s, {ntok/el/1e3:.0f}k tok/s, wait {wait/el:.2f})")
        if mac > best:
            best, bad = mac, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log(f"    조기 종료 (patience {C.PATIENCE}), 선택 epoch {best_ep}")
                break
    model.load_state_dict(best_state)
    preds = {}
    for nm, ds in (("val", va), ("test", te)):
        p = predict(model, model_name, ds, dev, bs)
        preds[nm] = pd.DataFrame(dict(sample_id=ds.sample_id, source_group=ds.source,
                                      source_subgroup=ds.subgroup,
                                      duplicate_group_id=ds.group, y_fp=ds.y, p_fp=p))
    return dict(model=model, history=pd.DataFrame(hist), best_epoch=best_ep,
                best_val_macro_auprc=best, preds=preds, a_max_dev=a_dev,
                n_params=sum(q.numel() for q in model.parameters()))

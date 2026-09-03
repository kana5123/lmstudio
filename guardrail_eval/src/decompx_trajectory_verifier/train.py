"""DXTV 학습 (§20-§21).  base/DecompX 는 캐시로만 등장하므로 학습 대상이 아니다."""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.evidence_cache import load_sample
from src.decompx_trajectory_verifier.model import DXTV


class CacheDataset(Dataset):
    """ragged 캐시 샤드들을 읽어 (C, f_attr, y_fp) 를 돌려준다."""

    def __init__(self, paths, split=None, attack_id=1, benign_id=0):
        self.items, self.blobs = [], []
        for p in paths:
            b = torch.load(p, weights_only=False)
            bi = len(self.blobs); self.blobs.append(b)
            for i in range(len(b["sample_id"])):
                if split is None or b["split"][i] == split:
                    self.items.append((bi, i))
        assert self.items, f"split={split} 에 표본이 없다"
        b0 = self.blobs[0]
        self.L, self.d, self.nC = b0["L"], b0["d"], b0["nC"]
        self.attack_id, self.benign_id = attack_id, benign_id
        self.y = torch.tensor([int(self.blobs[b]["y_fp"][i]) for b, i in self.items])
        self.cell = [self.blobs[b]["confusion_cell"][i] for b, i in self.items]
        self.source = [self.blobs[b]["source_group"][i] for b, i in self.items]
        self.sample_id = [self.blobs[b]["sample_id"][i] for b, i in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        bi, i = self.items[j]
        b = self.blobs[bi]
        Cx, Y, a, _ = load_sample(b, i)
        f = torch.stack([Y[:, self.benign_id], Y[:, self.attack_id], a], -1)   # [T,3]
        return Cx, f, float(self.y[j])

    def attr_stats(self):
        """§15 표준화 통계 -- 이 데이터셋(=train)에서만 구한다."""
        acc = []
        for j in range(len(self)):
            acc.append(self[j][1])
        x = torch.cat(acc, 0)
        return x.mean(0), x.std(0)


def collate(batch):
    """배치 안에서만 최대 길이로 패딩한다(고정 512 패딩 없음)."""
    Ls = [c.shape[0] for c, _, _ in batch]
    T = max(c.shape[1] for c, _, _ in batch)
    B, L, d = len(batch), Ls[0], batch[0][0].shape[2]
    Cx = torch.zeros(B, L, T, d)
    fa = torch.zeros(B, T, 3)
    mk = torch.zeros(B, T)
    y = torch.zeros(B)
    for i, (c, f, yy) in enumerate(batch):
        t = c.shape[1]
        Cx[i, :, :t] = c; fa[i, :t] = f; mk[i, :t] = 1.0; y[i] = yy
    return Cx, fa, mk, y


def make_perm(mask, generator=None):
    """A5 용 토큰 순열.  실토큰 안에서만 섞고 패딩 자리는 그대로 둔다."""
    B, T = mask.shape
    perm = torch.arange(T).unsqueeze(0).repeat(B, 1)
    for i in range(B):
        n = int(mask[i].sum())
        perm[i, :n] = torch.randperm(n, generator=generator)
    return perm


@torch.no_grad()
def evaluate(model, loader, dev, variant):
    model.eval()
    S, Y, tot, n = [], [], 0.0, 0
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    for Cx, fa, mk, y in loader:
        Cx, fa, mk, y = Cx.to(dev), fa.to(dev), mk.to(dev), y.to(dev)
        perm = make_perm(mk.cpu()).to(dev) if variant == "A5" else None
        lg = model(Cx, fa, mk, perm)
        tot += float(lossf(lg, y)); n += len(y)
        S.append(torch.sigmoid(lg).cpu()); Y.append(y.cpu())
    S, Y = torch.cat(S).numpy(), torch.cat(Y).numpy()
    m = dict(loss=tot / n, n=n, n_fp=int(Y.sum()))
    if len(np.unique(Y)) > 1:
        m |= dict(auroc=roc_auc_score(Y, S), auprc=average_precision_score(Y, S))
    else:
        m |= dict(auroc=float("nan"), auprc=float("nan"))
    return m, S, Y


def build_model(ds, variant, max_len=512, dev="cuda"):
    m = DXTV(ds.d, ds.L, max_len, variant=variant, d_v=C.D_V, depth_tf=C.DEPTH_TF,
             token_tf=C.TOKEN_TF, attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT,
             head_hidden=C.HEAD_HIDDEN).to(dev)
    if m.cfgv["use_attr"]:
        mu, sd = ds.attr_stats()
        m.anchor.set_stats(mu, sd)
    return m

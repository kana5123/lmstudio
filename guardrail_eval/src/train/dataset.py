"""검증기 학습용 특징 묶음.  **모든 정규화 통계는 ver_train 에서만 적합**한다.

한 표본이 갖는 것:
    global      : (768,)   delta_h = h_L - h_1  (또는 hL / h1 — 절제에 따라)
    delta_c     : (N,768)  토큰별 CLS 이동 기여
    directional : (N,)     dot(delta_c_k, v)
    margin      : (N,)     분류기 기여 차 (UNSAFE - BENIGN)
    mask        : (N,)     실토큰 여부
    numeric     : (2,)     [unsafe_probability, logit_margin]
    y           : 0/1      1=TP, 0=FP
"""
import glob, json, sys
from pathlib import Path

import numpy as np, torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "artifacts/features"
MAXN = 512


def load_decompx(split):
    """샤드를 합치고 sample_id -> 인덱스 사전을 만든다."""
    parts = sorted(glob.glob(str(FEAT / f"decompx_{split}_*of*.pt")))
    assert parts, f"decompx 특징이 없다: {split}"
    ds = [torch.load(p, weights_only=False) for p in parts]
    out = {k: torch.cat([d[k] for d in ds]) for k in
           ("delta_c", "directional", "margin", "mask", "input_ids",
            "gt", "unsafe_probability", "logit_margin", "seq_len", "recon_rel_err")}
    out["sample_id"] = [s for d in ds for s in d["sample_id"]]
    return out


def load_split(split, global_feat="delta_h", need_tokens=True):
    """주의: 행 순서는 **항상 sample_id 오름차순**으로 고정한다.

    예전에는 need_tokens 값에 따라 (은닉 파일 순서 / DecompX 파일 순서) 로 행 순서가
    달라졌다.  그러면 같은 절제를 --only 로 따로 돌릴 때와 전체로 돌릴 때 DataLoader
    셔플 결과가 달라져 **수치가 재현되지 않는다**.  순서를 sample_id 로 못박아 없앤다.
    """
    h = torch.load(FEAT / f"hidden_{split}.pt", weights_only=False)
    L = h["layers"]
    idx = {s: i for i, s in enumerate(h["sample_id"])}
    G = {"delta_h": h["h"][:, L] - h["h"][:, 1], "hL": h["h"][:, L], "h1": h["h"][:, 1],
         "concat": torch.cat([h["h"][:, 1], h["h"][:, L]], 1)}[global_feat]
    d = {"sample_id": h["sample_id"], "global": G, "y": h["gt"].float(),
         "numeric": torch.stack([h["unsafe_probability"], h["logit_margin"]], 1)}
    if need_tokens:
        dx = load_decompx(split)
        order = [idx[s] for s in dx["sample_id"]]
        # DecompX 쪽 순서에 맞춰 은닉 특징을 재배열 -> 두 원천의 sample_id 가 정확히 대응
        assert sorted(dx["sample_id"]) == sorted(h["sample_id"]), f"{split} 표본 집합 불일치"
        d = {"sample_id": [h["sample_id"][i] for i in order],
             "global": G[order], "y": h["gt"][order].float(),
             "numeric": torch.stack([h["unsafe_probability"], h["logit_margin"]], 1)[order]}
        assert (d["y"] == dx["gt"].float()).all(), f"{split} 라벨 불일치"
        d.update({k: dx[k] for k in ("delta_c", "directional", "margin", "mask",
                                     "seq_len", "recon_rel_err")})
    # 행 순서를 sample_id 로 고정 (need_tokens 와 무관하게 동일)
    srt = sorted(range(len(d["sample_id"])), key=lambda i: d["sample_id"][i])
    out = {"sample_id": [d["sample_id"][i] for i in srt]}
    for k, v in d.items():
        if k != "sample_id":
            out[k] = v[srt] if hasattr(v, "__getitem__") else v
    return out


class Scaler:
    """ver_train 에서만 적합. 다른 분할에는 transform 만 적용."""

    def fit(self, tr, need_tokens=True):
        g = tr["global"].float()
        self.g_mu, self.g_sd = g.mean(0), g.std(0).clamp_min(1e-6)
        n = tr["numeric"].float()
        self.n_mu, self.n_sd = n.mean(0), n.std(0).clamp_min(1e-6)
        if need_tokens:
            m = tr["mask"]
            self.c_sd = tr["delta_c"][m].float().std().clamp_min(1e-6)
            self.d_mu = tr["directional"][m].mean(); self.d_sd = tr["directional"][m].std().clamp_min(1e-6)
            self.m_mu = tr["margin"][m].mean();      self.m_sd = tr["margin"][m].std().clamp_min(1e-6)
        return self

    def state(self):
        return {k: v for k, v in self.__dict__.items()}


class VerifierDS(Dataset):
    def __init__(self, d, sc, need_tokens=True):
        self.d, self.sc, self.t = d, sc, need_tokens

    def __len__(self):
        return len(self.d["y"])

    def __getitem__(self, i):
        sc = self.sc
        o = {"global": (self.d["global"][i].float() - sc.g_mu) / sc.g_sd,
             "numeric": (self.d["numeric"][i].float() - sc.n_mu) / sc.n_sd,
             "y": self.d["y"][i]}
        if self.t:
            m = self.d["mask"][i]
            o["delta_c"] = self.d["delta_c"][i].float() / sc.c_sd
            o["directional"] = (self.d["directional"][i] - sc.d_mu) / sc.d_sd
            o["margin"] = (self.d["margin"][i] - sc.m_mu) / sc.m_sd
            o["mask"] = m
            for k in ("delta_c", "directional", "margin"):          # 패딩 자리 0 으로
                o[k] = o[k] * m.unsqueeze(-1) if o[k].dim() == 2 else o[k] * m
        return o

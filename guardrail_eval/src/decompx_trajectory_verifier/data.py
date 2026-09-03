"""§18 학습 I/O: memmap 스트리밍 + 토큰 길이 버킷팅 + 균형 표집.

전체 cache 를 RAM 에 올리지 않는다.  numpy memmap 으로 필요한 구간만 읽는다.
§3: a 는 캐시 값을 쓰지 않고 Y 차이로 매번 다시 계산하며, 캐시 값과 대조한다.
§5: dataset / source / split / cell / group / sample_id 는 verifier 입력에 넣지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import ART

MM = ART / "memmap"
A_TOL = 1e-4          # §3: a_runtime 과 a_cached 의 허용 차이


class MemmapEvidence:
    """memmap 핸들 묶음.  프로세스마다 하나만 만든다."""

    def __init__(self, root=MM):
        meta = json.load(open(root / "meta.json"))
        self.L, self.d = int(meta["L"]), int(meta["d"])
        self.C = np.load(root / "C.npy", mmap_mode="r")
        self.Y = np.load(root / "Y.npy", mmap_mode="r")
        self.a = np.load(root / "a.npy", mmap_mode="r")
        self.ids = np.load(root / "ids.npy", mmap_mode="r")
        self.index = pd.read_parquet(root / "index.parquet").set_index("sample_id")


class EvidenceDataset(Dataset):
    """manifest(sample_id, split, ...) 로 부분집합을 만든다."""

    def __init__(self, mm, manifest, split, attack_id=1, benign_id=0, need_C=True):
        self.mm = mm
        m = manifest[manifest.split == split].copy()
        # 메타데이터는 manifest 쪽을 쓰고, memmap index 에서는 위치/로짓만 가져온다
        j = m.join(mm.index[["offset", "length", "logit_0", "logit_1"]], on="sample_id")
        assert j.offset.notna().all(), "memmap 인덱스에 없는 sample_id 가 있다"
        self.off = j.offset.to_numpy()
        self.len = j.length.to_numpy()
        self.logits = j[["logit_0", "logit_1"]].to_numpy(np.float32)
        self.y = (j.confusion_cell.to_numpy() == "FP").astype(np.float32)
        self.source = j.source_group.to_numpy()
        self.subgroup = j.source_subgroup.to_numpy()
        self.group = j.duplicate_group_id.to_numpy()
        self.sample_id = j.sample_id.to_numpy()
        self.attack_id, self.benign_id, self.need_C = attack_id, benign_id, need_C
        self.a_max_dev = 0.0

    def __len__(self):
        return len(self.off)

    def __getitem__(self, i):
        o, n = int(self.off[i]), int(self.len[i])
        Y = torch.from_numpy(np.array(self.mm.Y[o:o + n]))                 # [T,2]
        a_run = Y[:, self.attack_id] - Y[:, self.benign_id]                  # §3 런타임 재계산
        a_cache = torch.from_numpy(np.array(self.mm.a[o:o + n]))
        dev = float((a_run - a_cache).abs().max()) if n else 0.0
        if dev > self.a_max_dev:
            self.a_max_dev = dev
        f = torch.stack([Y[:, self.benign_id], Y[:, self.attack_id], a_run], -1)   # [T,3]
        C = (torch.from_numpy(np.array(self.mm.C[o:o + n])).permute(1, 0, 2)
             if self.need_C else torch.zeros(1, n, 1))
        return C, f, torch.from_numpy(self.logits[i]), float(self.y[i]), dev


def collate(batch):
    Cs, fs, lg, ys, devs = zip(*batch)
    T = max(f.shape[0] for f in fs)
    B, L, d = len(batch), Cs[0].shape[0], Cs[0].shape[2]
    C = torch.zeros(B, L, T, d)
    fa = torch.zeros(B, T, 3)
    mk = torch.zeros(B, T)
    for i, (c, f) in enumerate(zip(Cs, fs)):
        t = f.shape[0]
        C[i, :, :t] = c; fa[i, :t] = f; mk[i, :t] = 1.0
    return C, fa, mk, torch.stack(lg), torch.tensor(ys), max(devs)


class BalancedBucketSampler(Sampler):
    """§11 source_group 균형 -> source 내부 TP/FP 균형.  그 안에서 길이 버킷으로 묶는다."""

    def __init__(self, ds, batch_size, seed=0, n_batches=None, bucket_mult=8):
        self.ds, self.bs, self.seed, self.bm = ds, batch_size, seed, bucket_mult
        w = np.zeros(len(ds))
        srcs = np.unique(ds.source)
        for s in srcs:
            ms = ds.source == s
            for c in (0.0, 1.0):
                m = ms & (ds.y == c)
                if m.sum():
                    w[m] = 1.0 / (len(srcs) * 2 * m.sum())
        self.w = w / w.sum()
        self.n_batches = n_batches or max(1, len(ds) // batch_size)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        self.epoch += 1
        n = self.n_batches * self.bs
        idx = rng.choice(len(self.ds), size=n, replace=True, p=self.w)
        # 길이 버킷: 큰 덩어리 안에서 길이순 정렬 후 배치로 자르면 패딩이 줄어든다
        chunk = self.bs * self.bm
        out = []
        for s in range(0, n, chunk):
            blk = idx[s:s + chunk]
            blk = blk[np.argsort(self.ds.len[blk])]
            for b in range(0, len(blk), self.bs):
                out.append(blk[b:b + self.bs].tolist())
        rng.shuffle(out)
        return iter(out)

    def __len__(self):
        return self.n_batches


class LengthBucketSampler(Sampler):
    """평가용: 자연 빈도 그대로, 길이순으로만 묶어 패딩을 줄인다."""

    def __init__(self, ds, batch_size):
        self.order = np.argsort(ds.len)
        self.bs = batch_size

    def __iter__(self):
        return iter([self.order[i:i + self.bs].tolist()
                     for i in range(0, len(self.order), self.bs)])

    def __len__(self):
        return int(np.ceil(len(self.order) / self.bs))

"""모든 은닉표현 파일을 하나의 긴 표로 모으고, 모델 간 공유되는 분할을 만든다.

분할을 모델 간 공유하는 이유: 같은 텍스트를 모델 A 에서는 학습에, 모델 B 에서는
평가에 쓰면 공유 검증기 입장에서는 그 텍스트를 이미 본 것이 된다.  중복 그룹
(duplicate_group_id) 단위로 한 번만 나누고 6개 모델에 동일하게 적용한다.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.shared_verifier.evidence import depth_resample, geom_features

ROOT = Path(__file__).resolve().parents[2]
HID = ROOT / "artifacts/shared_verifier/hidden"
TEST_FRAC = 0.30


def split_of(group_id, seed=0):
    """중복 그룹 id 를 해시해서 train/test 로 결정론적으로 배정."""
    h = hashlib.blake2b(f"{seed}:{group_id}".encode(), digest_size=8).digest()
    return "test" if int.from_bytes(h, "big") / 2**64 < TEST_FRAC else "train"


def load_all(seed=0):
    """-> (표, 특징 사전).  표는 행 하나가 (모델, 데이터셋, 샘플) 하나."""
    meta, F = [], {"geom": [], "raw_last": []}
    for f in sorted(HID.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        h = d["h_cls"]
        g, gnames = geom_features(h)
        F["geom"].append(g.astype(np.float32))
        F["raw_last"].append(depth_resample(h.float())[:, -1].numpy().astype(np.float32))
        cell = np.array(d["cell"])
        meta.append(pd.DataFrame(dict(
            model=d["model"], dataset=d["dataset"], attack_family=d["attack_family"],
            sample_id=d["sample_id"], dup=d["dup"], cell=cell,
            correct=np.isin(cell, ["TP", "TN"]).astype(np.int8),
            y=d["y"].numpy(), pred=d["pred"].numpy(), p_attack=d["p_attack"].numpy(),
        )))
    tab = pd.concat(meta, ignore_index=True)
    # 어휘 대조군용 원문.  sample_id -> text 는 모델과 무관하게 동일하다.
    src = pd.read_parquet(ROOT / "data/multisource_guard/canonical_samples.parquet",
                          columns=["sample_id", "text"])
    tab = tab.merge(src, on="sample_id", how="left")
    assert tab.text.notna().all(), "원문을 못 찾은 샘플이 있다"
    tab = tab.reset_index(drop=True)   # 특징 배열이 위치 인덱스라 0..n-1 이어야 한다
    tab["split"] = [split_of(g, seed) for g in tab.dup]
    feats = {k: np.concatenate(v, 0) for k, v in F.items()}
    feats["conf"] = np.c_[tab.p_attack.to_numpy(np.float32),
                          np.abs(tab.p_attack.to_numpy(np.float32) - 0.5)]
    return tab, feats, gnames


def check_no_leak(tab):
    """같은 중복 그룹이 두 분할에 걸치면 안 된다."""
    bad = tab.groupby("dup").split.nunique()
    assert (bad == 1).all(), f"분할 누수: 그룹 {int((bad > 1).sum())}개가 train/test 에 걸침"
    return int(bad.size)

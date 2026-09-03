"""§12 source 쌍 전이 행렬.  A0 / A3 만, 최소 학습.

한 source 에서만 학습/검증하고 세 source 의 test 부분에 각각 평가한다.
분할은 기존 seen_source_seed{N} manifest 를 그대로 쓴다(새 분할을 만들지 않는다).
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.train_c1 import run

SPLITS = ART / "phase_c1/split_manifests"
OUT = RES / "phase_d0"
SRC = {"WJ": "wildjailbreak:adversarial", "PS": "promptshield:test", "QS": "piguard:Question Set"}


def single_source_manifest(seed, train_src):
    m = pd.read_parquet(SPLITS / f"seen_source_seed{seed}.parquet").copy()
    keep_tr = (m.source_group == train_src) & m.split.isin(["train", "val"])
    keep_te = m.split == "test"                      # 세 source 의 test 를 모두 평가에 쓴다
    out = m[keep_tr | keep_te].copy()
    out.loc[keep_te & (out.source_group == train_src) & (out.split != "test"), "split"] = "test"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A0,A3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for mdl in a.models.split(","):
            for tag, src in SRC.items():
                man = single_source_manifest(seed, src)
                print(f"\n=== train {tag} / {mdl} / seed {seed}  "
                      f"(train {int((man.split=='train').sum())}, val {int((man.split=='val').sum())}, "
                      f"test {int((man.split=='test').sum())}) ===", flush=True)
                r = run(mdl, man, seed, bs=a.bs, log=lambda s: print(s, flush=True))
                te = r["preds"]["test"]
                for s2tag, s2 in SRC.items():
                    g = te[te.source_group == s2]
                    y, p = g.y_fp.to_numpy(), g.p_fp.to_numpy()
                    rows.append(dict(model=mdl, seed=seed, train_source=tag, test_source=s2tag,
                                     n=len(g), n_fp=int(y.sum()),
                                     auroc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                                     auprc=average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan))
                del r
                torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(OUT / f"pairwise_transfer_seed{a.seeds.replace(',','-')}.csv",
                                          index=False)
    df = pd.DataFrame(rows)
    for mdl in a.models.split(","):
        piv = df[df.model == mdl].pivot_table(index="train_source", columns="test_source",
                                              values="auroc")
        piv.to_csv(OUT / f"pairwise_transfer_{mdl.lower()}.csv")
        print(f"\n=== {mdl} AUROC 전이 행렬 (행=학습 source, 열=평가 source) ===")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()

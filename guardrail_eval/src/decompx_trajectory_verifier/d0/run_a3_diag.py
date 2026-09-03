"""§5, §7, §8, §9 실행."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import ART, RES
from src.decompx_trajectory_verifier.d0.a3_intervene import forward_intervened, fusion_split, load_a3
from src.decompx_trajectory_verifier.d0.probes import SHORT, SRC_ORDER, fp_tp_direction
from src.decompx_trajectory_verifier.data import (EvidenceDataset, LengthBucketSampler,
                                                  MemmapEvidence, collate)

OUT = RES / "phase_d0"
CK = ART / "phase_c1/checkpoints"
SPLITS = ART / "phase_c1/split_manifests"
PROTOCOLS = {"seen_source": "test", "loso_wj": "test", "loso_ps": "test", "loso_qs": "test"}


def eval_conditions(model, ds, dev, conds, bs=32):
    """조건별 예측을 한 번의 데이터 순회로 모두 계산한다."""
    sam = LengthBucketSampler(ds, bs)
    dl = DataLoader(ds, batch_sampler=sam, collate_fn=collate, num_workers=4)
    order, P = [], {k: [] for k in conds}
    for b, (Cx, fa, mk, lg, y, _) in zip(sam, dl):
        Cx, fa, mk = Cx.to(dev), fa.to(dev), mk.to(dev)
        for k, kw in conds.items():
            P[k].append(torch.sigmoid(forward_intervened(model, Cx, fa, mk, **kw)["logit"]).cpu())
        order += list(b)
    o = np.array(order)
    out = {}
    for k in conds:
        v = np.zeros(len(ds), np.float32); v[o] = torch.cat(P[k]).numpy(); out[k] = v
    return out


def rep_source_probe(X, meta_cell, meta_src, tr, te, cell):
    m = meta_cell == cell
    a, b = tr & m, te & m
    if len(np.unique(meta_src[a])) < 2 or len(np.unique(meta_src[b])) < 2:
        return None
    sc = StandardScaler().fit(X[a])
    c = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=8).fit(sc.transform(X[a]), meta_src[a])
    p = c.predict(sc.transform(X[b]))
    return dict(cell=cell, n_train=int(a.sum()), n_test=int(b.sum()),
                macro_f1=f1_score(meta_src[b], p, average="macro"),
                balanced_acc=balanced_accuracy_score(meta_src[b], p))


def main(seed=0, dev="cuda", bs=32):
    OUT.mkdir(parents=True, exist_ok=True)
    mm = MemmapEvidence()
    # ---------- §7 branch zeroing + §9 layer occlusion ----------------------
    CONDS = {"A_original": {}, "B_zero_trajectory": dict(zero_tau=True),
             "C_zero_attribution": dict(zero_attr=True),
             "D_zero_both": dict(zero_tau=True, zero_attr=True)}
    for l in range(12):
        CONDS[f"occ_L{l+1}"] = dict(occlude_layers=[l])
    CONDS["occ_early_L1-4"] = dict(occlude_layers=list(range(0, 4)))
    CONDS["occ_middle_L5-8"] = dict(occlude_layers=list(range(4, 8)))
    CONDS["occ_late_L9-12"] = dict(occlude_layers=list(range(8, 12)))

    rows, reps_rows, dir_rows, fw_rows = [], [], [], []
    for proto in PROTOCOLS:
        ck = CK / f"{proto}_A3_seed{seed}.pt"
        m = load_a3(ck, mm.d, mm.L, dev)
        fw_rows.append(dict(protocol=proto, seed=seed, **fusion_split(m)))
        man = pd.read_parquet(SPLITS / f"{proto}_seed{seed}.parquet")
        ds = EvidenceDataset(mm, man, "test")
        P = eval_conditions(m, ds, dev, CONDS, bs)
        y, g = ds.y, ds.group
        base = roc_auc_score(y, P["A_original"])
        for k, p in P.items():
            au = roc_auc_score(y, p)
            rows.append(dict(protocol=proto, seed=seed, condition=k, auroc=au,
                             auprc=average_precision_score(y, p),
                             delta_vs_original=au - base))
        # ---------- §5 표현 source probe + §6 방향 코사인 --------------------
        if proto == "seen_source":
            man_all = man.copy()
            reps = {k: [] for k in ("tau_mean", "v_mean", "vcls")}
            order = []
            for sp in ("train", "test"):
                d2 = EvidenceDataset(mm, man_all, sp)
                sam = LengthBucketSampler(d2, bs)
                dl = DataLoader(d2, batch_sampler=sam, collate_fn=collate, num_workers=4)
                idx, acc = [], {k: [] for k in reps}
                for b, (Cx, fa, mk, lg, yy, _) in zip(sam, dl):
                    o = forward_intervened(m, Cx.to(dev), fa.to(dev), mk.to(dev), want_reps=True)
                    for k in reps:
                        acc[k].append(o[k].cpu())
                    idx += list(b)
                o_ = np.array(idx)
                for k in reps:
                    v = torch.cat(acc[k]).numpy(); z = np.zeros_like(v); z[o_] = v
                    reps[k].append(z)
                order.append(pd.DataFrame(dict(sample_id=d2.sample_id, source_group=d2.source,
                                               confusion_cell=np.where(d2.y == 1, "FP", "TP"),
                                               split=sp)))
            mt = pd.concat(order, ignore_index=True)
            R = {k: np.concatenate(v, 0) for k, v in reps.items()}
            tr = (mt.split == "train").to_numpy(); te = (mt.split == "test").to_numpy()
            src = mt.source_group.to_numpy(); cell = mt.confusion_cell.to_numpy()
            for k, X in R.items():
                for c in ("TP", "FP"):
                    r = rep_source_probe(X, cell, src, tr, te, c)
                    if r:
                        reps_rows.append(dict(representation=k, seed=seed, **r))
                dir_rows.append(dict(seed=seed, **fp_tp_direction(X, mt, k)))
        del m
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(OUT / f"branch_and_occlusion_seed{seed}.csv", index=False)
    pd.DataFrame(reps_rows).to_csv(OUT / f"a3_rep_source_probe_seed{seed}.csv", index=False)
    pd.DataFrame(dir_rows).to_csv(OUT / f"a3_rep_direction_seed{seed}.csv", index=False)
    pd.DataFrame(fw_rows).to_csv(OUT / f"fusion_weight_audit_seed{seed}.csv", index=False)
    print("완료")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))

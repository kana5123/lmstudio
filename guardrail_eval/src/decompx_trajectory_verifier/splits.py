"""PHASE C0: seen-source / Leave-One-Source-Out split manifest 생성 (§6-§8).

분할 단위는 duplicate_group_id 다.  같은 그룹은 한 split 에만 들어간다.
(source_group, confusion_cell) 층 안에서 배정하므로 각 split 이 TP·FP 를 모두 갖는다.
held-out source 의 라벨은 학습/조기종료/하이퍼파라미터/정규화에 쓰지 않는다.
"""
import hashlib, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import ART, DATA, RES

SPLITDIR = ART / "phase_c1/split_manifests"
SOURCES = ["wildjailbreak:adversarial", "promptshield:test", "piguard:Question Set"]
FOLDS = {"wj": "wildjailbreak:adversarial", "ps": "promptshield:test",
         "qs": "piguard:Question Set"}
KEEP = ["sample_id", "duplicate_group_id", "source_group", "source_subgroup",
        "confusion_cell", "split"]


def _h(seed, tag, g):
    return int.from_bytes(hashlib.blake2b(f"{seed}|{tag}|{g}".encode(), digest_size=8).digest(),
                          "big") / 2 ** 64


def stratified_group_split(df, seed, fracs, tag=""):
    """(source_group, confusion_cell) 층별로 그룹을 정렬해 비율대로 자른다."""
    names = list(fracs)
    out = {}
    for (src, cell), sub in df.groupby(["source_group", "confusion_cell"]):
        groups = sorted(sub.duplicate_group_id.unique(), key=lambda g: _h(seed, f"{tag}|{src}|{cell}", g))
        n = len(groups)
        edges, acc = [], 0.0
        for k in names[:-1]:
            acc += fracs[k]; edges.append(int(round(n * acc)))
        bounds = [0] + edges + [n]
        for j, k in enumerate(names):
            for g in groups[bounds[j]:bounds[j + 1]]:
                out.setdefault(g, k)          # 다른 cell 에서 이미 배정됐으면 유지
    s = df.duplicate_group_id.map(out)
    assert s.notna().all()
    return s


def build_seen(core, seed):
    d = core.copy()
    d["split"] = stratified_group_split(d, seed, dict(train=.70, val=.15, test=.15), "seen")
    return d[KEEP]


def build_loso(core, fold, seed):
    held = FOLDS[fold]
    tr_src = core[core.source_group != held].copy()
    te_src = core[core.source_group == held].copy()
    # 학습 source 안에서만 group-aware train/val (70:15 비율 유지 -> 0.824:0.176)
    tr_src["split"] = stratified_group_split(tr_src, seed, dict(train=.824, val=.176),
                                             f"loso_{fold}")
    te_src["split"] = "test"
    return pd.concat([tr_src, te_src], ignore_index=True)[KEEP]


def audit(man, protocol, fold):
    rows, ok = [], True
    for split in ("train", "val", "test"):
        s = man[man.split == split]
        for src, g in s.groupby("source_group"):
            c = g.confusion_cell.value_counts()
            rows.append(dict(protocol=protocol, fold=fold, split=split, source=src,
                             TP=int(c.get("TP", 0)), FP=int(c.get("FP", 0)),
                             groups=g.duplicate_group_id.nunique(), n=len(g)))
    # 겹침 검사
    sets = {k: set(man[man.split == k].sample_id) for k in ("train", "val", "test")}
    gsets = {k: set(man[man.split == k].duplicate_group_id) for k in ("train", "val", "test")}
    ov = {f"{a}∩{b}": len(sets[a] & sets[b]) for a, b in
          (("train", "val"), ("train", "test"), ("val", "test"))}
    gov = {f"{a}∩{b}": len(gsets[a] & gsets[b]) for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))}
    leak = 0
    if protocol == "loso":
        held = FOLDS[fold]
        leak = int(((man.source_group == held) & (man.split != "test")).sum())
    ok = all(v == 0 for v in ov.values()) and all(v == 0 for v in gov.values()) and leak == 0
    # TP/FP 존재 검사 (LOSO train/val 은 held-out 이 없으므로 학습 source 만)
    for r in rows:
        if protocol == "seen" or r["split"] == "test" or True:
            ok &= (r["TP"] > 0 and r["FP"] > 0)
    return rows, dict(protocol=protocol, fold=fold, sample_overlap=ov, group_overlap=gov,
                      heldout_leakage=leak, ok=bool(ok))


def main(seeds=(0, 1, 2, 3, 4)):
    core = pd.read_parquet(DATA / "core_tp_fp.parquet")
    SPLITDIR.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for seed in seeds:
        m = build_seen(core, seed)
        m.to_parquet(SPLITDIR / f"seen_source_seed{seed}.parquet", index=False)
        r, c = audit(m, "seen", "-"); c["seed"] = seed
        rows += [dict(seed=seed, **x) for x in r]; checks.append(c)
        for fold in FOLDS:
            m = build_loso(core, fold, seed)
            m.to_parquet(SPLITDIR / f"loso_{fold}_seed{seed}.parquet", index=False)
            r, c = audit(m, "loso", fold); c["seed"] = seed
            rows += [dict(seed=seed, **x) for x in r]; checks.append(c)
    A = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "phase_c1").mkdir(exist_ok=True)
    A.to_csv(RES / "phase_c1/split_audit.csv", index=False)
    return A, pd.DataFrame(checks)


if __name__ == "__main__":
    A, Ck = main()
    print("=== split 누수 감사 (모든 protocol x fold x seed) ===")
    bad = Ck[~Ck.ok]
    print(f"  검사 {len(Ck)}건, 실패 {len(bad)}건")
    print(f"  표본 겹침 최대 {max(max(c.values()) for c in Ck.sample_overlap)}, "
          f"그룹 겹침 최대 {max(max(c.values()) for c in Ck.group_overlap)}, "
          f"held-out 누수 최대 {Ck.heldout_leakage.max()}")
    if len(bad):
        print(bad.to_string()); sys.exit(1)
    print("\n=== seed 0 split 구성 ===")
    s0 = A[A.seed == 0]
    for proto, fold in [("seen", "-"), ("loso", "wj"), ("loso", "ps"), ("loso", "qs")]:
        t = s0[(s0.protocol == proto) & (s0.fold == fold)]
        print(f"\n[{proto} {fold}]")
        print(t[["split", "source", "TP", "FP", "groups", "n"]].to_string(index=False))
    print(f"\n저장 -> {SPLITDIR} (5 seed x 4 protocol = 20 manifest)")

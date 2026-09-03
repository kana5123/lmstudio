"""cross-dataset delta_CORR cos 의 치환 귀무 + 전이 AUROC 의 치환 귀무.

L11->L12 는 이동벡터의 유효랭크가 1.45 로 극히 낮아, 아무 방향이나 정렬되기 쉽다.
따라서 관측 cos 를 **실측 귀무**와 비교해야 한다.

치환: 두 데이터셋 각각의 TRAIN 에서, 정답 라벨 층화를 보존한 채 정오 배정만 섞는다.
      (correctness_direction.perm_labels 와 동일 설계)
"""
import itertools, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direction_correctness.correctness_direction import (
    load, unit, group_split, cell_means, directions, perm_labels, auroc,
    CELLS, CORRECT, ATTACK, SEEDS, MIN_CELL_TEST)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/direction_correctness"
NPERM = 500


def main(tag="move"):
    D = load()
    X_all = D["g"] if tag == "move" else D["h"]
    L = X_all.shape[1]
    names = ([f"L{li}->L{li+1}" for li in range(L)] if tag == "move" else [f"L{li}" for li in range(L)])
    data, obs_dir, sp = {}, {}, {}
    for ds in sorted(set(D["dataset"])):
        m = D["dataset"] == ds
        Xd, cd, dupd = X_all[m], D["cell"][m], D["dup"][m]
        if min(int((cd == c).sum()) for c in CELLS) < 40:
            continue
        tr, te = group_split(dupd, cd, 0)
        if min(int((cd[te] == c).sum()) for c in CELLS) < MIN_CELL_TEST:
            continue
        data[ds] = (Xd, cd, tr, te)
        obs_dir[ds] = directions(cell_means(Xd[tr], cd[tr]))
    dss = sorted(data)
    print(f"repr={tag}  데이터셋 {dss}  치환 {NPERM}회 (최소 도달 p={1/(NPERM+1):.2e})")

    # 치환 방향 미리 생성
    perm_dir = {}
    for ds in dss:
        Xd, cd, tr, te = data[ds]
        rng = np.random.default_rng(7)
        arr = np.empty((NPERM, L, Xd.shape[2]), dtype=np.float32)
        for b in range(NPERM):
            cp = perm_labels(cd[tr], rng)
            arr[b] = directions(cell_means(Xd[tr], cp))["CORR"]
        perm_dir[ds] = arr
        print(f"  {ds} 치환 방향 생성", flush=True)

    rows = []
    for a, b in itertools.combinations(dss, 2):
        na = unit(perm_dir[a].astype(np.float64)); nb = unit(perm_dir[b].astype(np.float64))
        nc = np.einsum("blh,blh->bl", na, nb)
        for li in range(L):
            o = float(np.dot(unit(obs_dir[a]["CORR"][li]), unit(obs_dir[b]["CORR"][li])))
            nn = nc[:, li]
            p1 = ((np.sum(nn >= o) + 1) / (NPERM + 1)) if o >= 0 else ((np.sum(nn <= o) + 1) / (NPERM + 1))
            rows.append({"pair": f"{a}|{b}", "repr": tag, "layer": names[li], "cos": o,
                         "null_mean": float(nn.mean()), "null_std": float(nn.std()),
                         "null_p95": float(np.percentile(nn, 95)),
                         "signed_p": float(p1), "min_attainable_p": 1 / (NPERM + 1)})
    cd_ = pd.DataFrame(rows)
    cd_.to_csv(RES / f"crossdataset_cosine_null__{tag}.csv", index=False)
    print("\n=== cross-dataset cos(delta_CORR) vs 실측 귀무 ===")
    print(f"{'쌍':42} {'층':10} {'cos':>7} {'귀무평균':>8} {'귀무p95':>8} {'p':>7}")
    for _, r in cd_.iterrows():
        if r.layer in ("L0->L1", "L0"):
            continue
        flag = "★" if r.signed_p <= 0.05 else " "
        if r.layer in (names[-1], names[-2], names[-4]):
            print(f"{r.pair:42} {r.layer:10} {r.cos:+7.3f} {r.null_mean:8.3f} "
                  f"{r.null_p95:8.3f} {r.signed_p:7.4f}{flag}")

    # 전이 AUROC 귀무
    rows = []
    for a in dss:
        for b in dss:
            if a == b:
                continue
            Xb, cb, trb, teb = data[b]
            y = np.isin(cb[teb], CORRECT).astype(int)
            for li in range(L):
                o = auroc(y, Xb[teb][:, li] @ unit(obs_dir[a]["CORR"][li]))
                nn = np.array([auroc(y, Xb[teb][:, li] @ unit(perm_dir[a][k, li].astype(np.float64)))
                               for k in range(0, NPERM, 5)])       # 100개로 축약
                p1 = (np.sum(nn >= o) + 1) / (len(nn) + 1)
                rows.append({"fit": a, "test": b, "repr": tag, "layer": names[li],
                             "auroc": o, "null_mean": float(nn.mean()),
                             "null_p95": float(np.percentile(nn, 95)), "signed_p": float(p1),
                             "n_null": len(nn), "min_attainable_p": 1 / (len(nn) + 1)})
    td = pd.DataFrame(rows)
    td.to_csv(RES / f"crossdataset_transfer_null__{tag}.csv", index=False)
    print("\n=== 전이 correctness AUROC vs 귀무 (마지막 층) ===")
    for _, r in td[td.layer == names[-1]].iterrows():
        flag = "★" if r.signed_p <= 0.05 else " "
        print(f"  {r.fit:20} -> {r['test']:20} AUROC={r.auroc:.3f} "
              f"귀무 {r.null_mean:.3f} (p95 {r.null_p95:.3f})  p={r.signed_p:.4f}{flag}")
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "move")

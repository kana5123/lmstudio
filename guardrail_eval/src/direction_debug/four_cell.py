"""§4~§9 — 2x2 혼동 셀 분해, 효과 방향, 치환 귀무, 신뢰도 대조.

정의 (모두 CORRECT - INCORRECT 방향으로 통일):
  delta_U    = mu_TP - mu_FP          UNSAFE 가지에서 맞은 것 - 틀린 것
  delta_S    = mu_TN - mu_FN          SAFE   가지에서 맞은 것 - 틀린 것
  delta_GT   = 0.5[(TP+FN) - (FP+TN)] 정답 클래스 (공격 - 정상), 예측가지 균형
  delta_PRED = 0.5[(TP+FP) - (FN+TN)] 모델 예측 (UNSAFE - SAFE), 정답 균형
  delta_CORR = 0.5[(TP+TN) - (FP+FN)] 정오 상호작용 (맞음 - 틀림), 양쪽 균형

**cell 표본 수가 크게 다르므로 pooled mean 을 쓰지 않고 각 cell 평균에 동일 가중을 준다.**

치환 귀무(§3): 이론값 1/sqrt(768) 을 쓰지 않는다.  같은 데이터에서
  - 예측 가지를 보존한 채 정오 라벨만 섞고 (cos(U,S) 용)
  - cell 라벨을 섞고 (cross-dataset 용)
방향을 다시 만들어 **부호 있는(signed)** cos 분포를 저장한다.  N=10,000.
따라서 도달 가능한 최소 p 값은 1/(N+1) = 9.999e-05 이며 그보다 작은 p 는 주장하지 않는다.
"""
import glob, itertools, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/direction_debug"
RES = ROOT / "results/direction_debug"
NPERM = 10000
NBOOT = 2000
EPS = 1e-12
CELLS = ("TP", "FP", "TN", "FN")


def load():
    fs = sorted(glob.glob(str(ART / "cellhidden_*of*.pt")))
    assert fs, "cell 은닉표현 없음"
    ds = [torch.load(f, weights_only=False) for f in fs]
    out = {"h": torch.cat([d["h"] for d in ds]).numpy().astype(np.float64),
           "logit_unsafe": torch.cat([d["logit_unsafe"] for d in ds]).numpy(),
           "logit_benign": torch.cat([d["logit_benign"] for d in ds]).numpy(),
           "text_len": torch.cat([d["text_len"] for d in ds]).numpy()}
    for k in ("sample_id", "dataset", "cell", "split_role", "dup"):
        out[k] = np.array([x for d in ds for x in d[k]])
    out["g"] = out["h"][:, 1:] - out["h"][:, :-1]
    return out


def cell_means(G, cell):
    """G:(n,L,H), cell:(n,) -> dict cell -> (L,H), 그리고 개수."""
    mu, cnt = {}, {}
    for c in CELLS:
        m = cell == c
        cnt[c] = int(m.sum())
        mu[c] = G[m].mean(0) if m.sum() > 0 else None
    return mu, cnt


def contrasts(mu):
    """모두 CORRECT-INCORRECT 방향.  cell 이 하나라도 비면 None."""
    if any(mu[c] is None for c in CELLS):
        return None
    TP, FP, TN, FN = mu["TP"], mu["FP"], mu["TN"], mu["FN"]
    return {
        "U": TP - FP,
        "S": TN - FN,
        "GT": 0.5 * ((TP + FN) - (FP + TN)),
        "PRED": 0.5 * ((TP + FP) - (FN + TN)),
        "CORR": 0.5 * ((TP + TN) - (FP + FN)),
    }


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def cosl(a, b):
    return np.einsum("lh,lh->l", unit(a), unit(b))


def wls_check(G, cell, layer):
    """§7 — 2x2 요인 가중최소제곱.  각 cell 이 동일 총가중을 갖게 한다.
    beta_CORR 방향이 delta_CORR 와 일치해야 한다(구현 교차검증)."""
    gt = np.where(np.isin(cell, ["TP", "FN"]), 1.0, -1.0)      # 공격 +1
    pr = np.where(np.isin(cell, ["TP", "FP"]), 1.0, -1.0)      # UNSAFE +1
    X = np.stack([np.ones_like(gt), gt, pr, gt * pr], 1)       # (n,4)
    w = np.zeros(len(cell))
    for c in CELLS:
        m = cell == c
        if m.sum():
            w[m] = 1.0 / m.sum()                                # cell 당 총가중 1
    Xw = X * w[:, None]
    beta = np.linalg.lstsq(Xw.T @ X, Xw.T @ G[:, layer], rcond=None)[0]   # (4,H)
    return beta[3]                                              # beta_CORR


def perm_null_US(G, cell, n=NPERM, seed=0):
    """cos(delta_U, delta_S) 귀무: **예측 가지를 보존**한 채 정오 라벨만 섞는다.
    즉 UNSAFE 가지 안에서 TP/FP 를, SAFE 가지 안에서 TN/FN 를 개수 보존해 섞는다."""
    rng = np.random.default_rng(seed)
    uns = np.isin(cell, ["TP", "FP"]); saf = ~uns
    iu, is_ = np.flatnonzero(uns), np.flatnonzero(saf)
    n_tp = int((cell == "TP").sum()); n_tn = int((cell == "TN").sum())
    L = G.shape[1]
    out = np.empty((n, L))
    for b in range(n):
        pu = rng.permutation(iu); ps = rng.permutation(is_)
        mu_tp = G[pu[:n_tp]].mean(0); mu_fp = G[pu[n_tp:]].mean(0)
        mu_tn = G[ps[:n_tn]].mean(0); mu_fn = G[ps[n_tn:]].mean(0)
        out[b] = cosl(mu_tp - mu_fp, mu_tn - mu_fn)
    return out


def perm_null_cell(G, cell, n=NPERM, seed=0):
    """cell 라벨 전체를 개수 보존해 섞어 각 대비 방향의 귀무를 만든다."""
    rng = np.random.default_rng(seed)
    L = G.shape[1]
    keys = ("U", "S", "GT", "PRED", "CORR")
    store = {k: np.empty((n, L, G.shape[2])) for k in keys}
    for b in range(n):
        c = rng.permutation(cell)
        mu, _ = cell_means(G, c)
        d = contrasts(mu)
        for k in keys:
            store[k][b] = d[k]
    return store


def signed_p(obs, null):
    """부호 있는 p: 관측이 양수면 P(null >= obs), 음수면 P(null <= obs)."""
    n = len(null)
    if obs >= 0:
        p = (np.sum(null >= obs) + 1) / (n + 1)
    else:
        p = (np.sum(null <= obs) + 1) / (n + 1)
    two = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return float(p), float(two)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    D = load()
    L = D["g"].shape[1]
    trans = [f"L{li}->L{li+1}" for li in range(L)]
    datasets = sorted(set(D["dataset"]))
    print(f"층 전이 {L}개, 데이터셋 {datasets}")

    dirs = {}          # (ds) -> {key: (L,H)}
    rows_cnt, rows_branch, rows_eff, rows_norm, rows_conf = [], [], [], [], []
    for ds in datasets:
        m = (D["dataset"] == ds) & (D["split_role"] == "train")
        G, cell = D["g"][m], D["cell"][m]
        mu, cnt = cell_means(G, cell)
        rows_cnt.append({"dataset": ds, "split_role": "train", **cnt})
        d = contrasts(mu)
        if d is None:
            print(f"  [건너뜀] {ds}: 빈 cell 있음 {cnt}")
            continue
        dirs[ds] = d
        print(f"\n=== {ds}  train cell: " + " ".join(f"{c}={cnt[c]}" for c in CELLS) + " ===")

        # §7 구현 교차검증
        chk = [float(np.dot(unit(wls_check(G, cell, li)), unit(d["CORR"][li]))) for li in range(L)]
        print(f"  §7 WLS 교차검증 cos(beta_CORR, delta_CORR) 최소={min(chk):.6f} "
              f"(1.0 이어야 함)")

        # 크기
        for li in range(L):
            rows_norm.append({"dataset": ds, "transition": trans[li],
                              **{f"norm_{k}": float(np.linalg.norm(d[k][li])) for k in d}})

        # §5 branch 일치 + 귀무 + 부트스트랩
        nu = perm_null_US(G, cell)
        obs = cosl(d["U"], d["S"])
        rng = np.random.default_rng(0)
        for li in range(L):
            p1, p2 = signed_p(obs[li], nu[:, li])
            bs = []
            for _ in range(NBOOT):
                idx = {c: rng.choice(np.flatnonzero(cell == c), cnt[c], True) for c in CELLS}
                mm = {c: G[idx[c], li].mean(0) for c in CELLS}
                bs.append(float(np.dot(unit(mm["TP"] - mm["FP"]), unit(mm["TN"] - mm["FN"]))))
            rows_branch.append({"dataset": ds, "transition": trans[li],
                                "cos_U_S": float(obs[li]),
                                "null_mean": float(nu[:, li].mean()),
                                "null_std": float(nu[:, li].std()),
                                "null_p95": float(np.percentile(nu[:, li], 95)),
                                "null_p99": float(np.percentile(nu[:, li], 99)),
                                "signed_p": p1, "two_sided_p": p2,
                                "min_attainable_p": 1 / (NPERM + 1),
                                "boot_ci_lo": float(np.percentile(bs, 2.5)),
                                "boot_ci_hi": float(np.percentile(bs, 97.5))})

        # §8 효과 정체
        for li in range(L):
            rows_eff.append({"dataset": ds, "transition": trans[li],
                             "cos_U_GT": float(cosl(d["U"], d["GT"])[li]),
                             "cos_U_PRED": float(cosl(d["U"], d["PRED"])[li]),
                             "cos_U_CORR": float(cosl(d["U"], d["CORR"])[li]),
                             "cos_U_S": float(obs[li]),
                             "cos_S_CORR": float(cosl(d["S"], d["CORR"])[li]),
                             "cos_GT_CORR": float(cosl(d["GT"], d["CORR"])[li])})

        # §9 신뢰도 상관 (진단·통제 전용)
        mh = (D["dataset"] == ds) & (D["split_role"] == "heldout")
        Gh = D["g"][mh]
        lu, lm = D["logit_unsafe"][mh], D["logit_unsafe"][mh] - D["logit_benign"][mh]
        pu = 1 / (1 + np.exp(-lm))
        for k in ("U", "S", "GT", "PRED", "CORR"):
            for li in range(L):
                q = Gh[:, li] @ unit(d[k][li])
                rows_conf.append({"dataset": ds, "direction": k, "transition": trans[li],
                                  "spearman_logit_unsafe": float(spearmanr(q, lu).statistic),
                                  "spearman_logit_margin": float(spearmanr(q, lm).statistic),
                                  "spearman_unsafe_prob": float(spearmanr(q, pu).statistic)})

    pd.DataFrame(rows_cnt).to_csv(RES / "cell_counts_train.csv", index=False)
    pd.DataFrame(rows_branch).to_csv(RES / "branch_alignment.csv", index=False)
    pd.DataFrame(rows_eff).to_csv(RES / "effect_alignment_by_layer.csv", index=False)
    pd.DataFrame(rows_norm).to_csv(RES / "direction_norms.csv", index=False)
    pd.DataFrame(rows_conf).to_csv(RES / "confidence_correlations.csv", index=False)
    np.savez_compressed(ART / "directions.npz",
                        **{f"{ds}|{k}": v for ds, d in dirs.items() for k, v in d.items()})
    print(f"\n저장 -> {RES}, {ART/'directions.npz'}")


if __name__ == "__main__":
    main()

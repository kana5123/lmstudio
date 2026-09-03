"""전역 방향 정렬 분석 (지시문 6·9·10·11·12·13절).

DecompX 없이, **일반 은닉표현만으로** 다음을 한다.

  g_i^(l) = h_CLS,i^(l) - h_CLS,i^(l-1)          실제 CLS 이동 벡터
  mu_TP^(l), mu_FP^(l)                           ver_train 에서만 계산
  v_U^(l) = normalize(mu_TP^(l) - mu_FP^(l))     경험적 TP-vs-FP 평균차 방향
  tau^(l) = dot(v_U^(l), (mu_TP+mu_FP)/2)        중심 중점 사영
  p = dot(v,g),  q = p - tau,  cos = dot(v,g)/||g||

주의(지시문 10절): v_U 는 이 단계에서 "정확성 방향(correctness direction)"이 아니다.
학습 데이터에서 FP 평균 이동 -> TP 평균 이동을 잇는 방향일 뿐이며,
held-out 에서 같은 정렬이 재현되어야만 정확성 관련 신호라고 말할 수 있다.

주 분석 구간은 l = 2..L (층 l-1 -> l).  임베딩->1층은 별도 표시만 하고 주 분석에서 뺀다.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score

from data.splits import _dedup_rows, sid

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "artifacts/features"
RES = ROOT / "results/directional_alignment"
ART = ROOT / "artifacts/directional_alignment"
TRAIN, HELD = "ver_train", ("ver_dev", "eval_val", "eval_test")
EPS = 1e-12
NBOOT = 2000


def load(split):
    d = torch.load(FEAT / f"hidden_{split}.pt", weights_only=False)
    return d


def sources():
    return {sid(r["prompt"]): r["source"] for r in _dedup_rows()}


def movements(d):
    """g^(l) = h[:, l] - h[:, l-1].  반환 (n, L, H) — 색인 0 이 l=1(임베딩->1층)."""
    h = d["h"].numpy().astype(np.float64)
    return h[:, 1:] - h[:, :-1]


def boot_auroc(y, s, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[i]
        if len(set(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, s[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + EPS))


def fit_directions(G, y):
    """ver_train 전용.  G:(n,L,H) -> v,(L,H) / mu_tp / mu_fp / tau,(L,)"""
    tp, fp = G[y == 1], G[y == 0]
    mu_tp, mu_fp = tp.mean(0), fp.mean(0)                    # (L,H)
    dmu = mu_tp - mu_fp
    v = dmu / (np.linalg.norm(dmu, axis=-1, keepdims=True) + EPS)
    tau = np.einsum("lh,lh->l", v, (mu_tp + mu_fp) / 2)
    return v, mu_tp, mu_fp, tau


def project(G, v, tau):
    p = np.einsum("lh,nlh->nl", v, G)                        # (n,L)
    q = p - tau[None, :]
    cos = p / (np.linalg.norm(G, axis=-1) + EPS)
    return p, q, cos


def stats_rows(split, y, p, q, cos, L, tag=""):
    rows = []
    for li in range(L):
        l = li + 1                                            # 전이 l-1 -> l
        r = {"split": split, "subset": tag or "all", "transition": f"L{l-1}->L{l}",
             "layer_to": l, "main_analysis": bool(l >= 2),
             "n_tp": int((y == 1).sum()), "n_fp": int((y == 0).sum())}
        for nm, arr in (("p", p[:, li]), ("q", q[:, li]), ("cos", cos[:, li])):
            a, b = arr[y == 1], arr[y == 0]
            r[f"{nm}_tp_mean"] = float(a.mean()) if len(a) else float("nan")
            r[f"{nm}_fp_mean"] = float(b.mean()) if len(b) else float("nan")
            r[f"{nm}_tp_median"] = float(np.median(a)) if len(a) else float("nan")
            r[f"{nm}_fp_median"] = float(np.median(b)) if len(b) else float("nan")
            r[f"{nm}_cohens_d"] = cohens_d(a, b)
            if len(set(y)) > 1:
                r[f"{nm}_auroc"] = float(roc_auc_score(y, arr))
                r[f"{nm}_auprc"] = float(average_precision_score(y, arr))
            else:
                r[f"{nm}_auroc"] = r[f"{nm}_auprc"] = float("nan")
        lo, hi = boot_auroc(y, q[:, li])
        r["q_auroc_ci_lo"], r["q_auroc_ci_hi"] = lo, hi
        rows.append(r)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    SRC = sources()
    data = {s: load(s) for s in (TRAIN,) + HELD}
    G = {s: movements(d) for s, d in data.items()}
    Y = {s: d["gt"].numpy() for s, d in data.items()}
    SRCTOP = {s: np.array([SRC[i].split("/")[0] for i in d["sample_id"]])
              for s, d in data.items()}
    L = G[TRAIN].shape[1]
    print(f"층 전이 수 L={L} (색인 0 = 임베딩->1층, 주 분석은 L1->L2 부터)")
    for s in (TRAIN,) + HELD:
        print(f"  {s:10} n={len(Y[s]):5} TP={int((Y[s]==1).sum()):5} FP={int((Y[s]==0).sum()):5}")

    # ---------- A. 전역 진단 방향 (말뭉치 교란 가능) ----------
    v, mu_tp, mu_fp, tau = fit_directions(G[TRAIN], Y[TRAIN])
    torch.save({"v": torch.tensor(v), "tau": torch.tensor(tau),
                "fit_split": TRAIN, "kind": "GLOBAL-DIAGNOSTIC",
                "n_tp": int((Y[TRAIN] == 1).sum()), "n_fp": int((Y[TRAIN] == 0).sum()),
                "transitions": [f"L{l}->L{l+1}" for l in range(L)],
                "source_counts_tp": dict(Counter(SRCTOP[TRAIN][Y[TRAIN] == 1])),
                "source_counts_fp": dict(Counter(SRCTOP[TRAIN][Y[TRAIN] == 0]))},
               ART / "v_u.pt")
    torch.save(torch.tensor(mu_tp), ART / "mu_tp.pt")
    torch.save(torch.tensor(mu_fp), ART / "mu_fp.pt")

    fit_rows = []
    for li in range(L):
        fit_rows.append({"transition": f"L{li}->L{li+1}", "layer_to": li + 1,
                         "main_analysis": bool(li + 1 >= 2),
                         "norm_mu_tp": float(np.linalg.norm(mu_tp[li])),
                         "norm_mu_fp": float(np.linalg.norm(mu_fp[li])),
                         "centroid_l2": float(np.linalg.norm(mu_tp[li] - mu_fp[li])),
                         "cos_mu_tp_mu_fp": float(mu_tp[li] @ mu_fp[li] /
                                                  (np.linalg.norm(mu_tp[li]) * np.linalg.norm(mu_fp[li]) + EPS)),
                         "tau": float(tau[li])})
    write_csv(RES / "direction_fit_summary.csv", fit_rows)

    all_rows = []
    for s in (TRAIN,) + HELD:
        p, q, cos = project(G[s], v, tau)
        rows = stats_rows(s, Y[s], p, q, cos, L)
        all_rows += rows
        if s in ("eval_val", "eval_test"):
            write_csv(RES / f"global_alignment_{s}.csv", rows)
        np.savez_compressed(ART / f"proj_{s}.npz", p=p, q=q, cos=cos, y=Y[s],
                            sample_id=np.array(data[s]["sample_id"]), src=SRCTOP[s])
    write_csv(RES / "global_alignment_all_splits.csv", all_rows)

    print("\n=== 전역 진단 방향 (말뭉치 교란 가능, 주 증거 아님) — q 기준 ===")
    print(f"{'split':10} {'전이':10} {'TP q평균':>10} {'FP q평균':>10} {'AUROC':>7} "
          f"{'95%CI':>18} {'d':>7}")
    for r in all_rows:
        if r["layer_to"] < 2 or r["split"] == TRAIN:
            continue
        print(f"{r['split']:10} {r['transition']:10} {r['q_tp_mean']:10.3f} {r['q_fp_mean']:10.3f} "
              f"{r['q_auroc']:7.4f} [{r['q_auroc_ci_lo']:.4f},{r['q_auroc_ci_hi']:.4f}] "
              f"{r['q_cohens_d']:7.3f}")

    # ---------- B. 같은 출처(same-source) 통제 ----------
    print("\n=== 같은 출처 통제: 출처별 TP/FP 표본 수 ===")
    feas = {}
    for s in (TRAIN,) + HELD:
        cnt = {}
        for src in sorted(set(SRCTOP[s])):
            m = SRCTOP[s] == src
            cnt[src] = (int((Y[s][m] == 1).sum()), int((Y[s][m] == 0).sum()))
        feas[s] = cnt
        print(f"  {s:10} " + "  ".join(f"{k}=TP{a}/FP{b}" for k, (a, b) in cnt.items()))
    (RES / "same_source_counts.json").write_text(json.dumps(feas, ensure_ascii=False, indent=1))

    ss_rows = []
    for src in sorted(set(SRCTOP[TRAIN])):
        m = SRCTOP[TRAIN] == src
        ntp, nfp = int((Y[TRAIN][m] == 1).sum()), int((Y[TRAIN][m] == 0).sum())
        if ntp < 20 or nfp < 20:
            print(f"  [건너뜀] {src}: ver_train TP={ntp} FP={nfp} — 방향을 억지로 만들지 않는다")
            continue
        vs, mtp, mfp, ts = fit_directions(G[TRAIN][m], Y[TRAIN][m])
        for s in HELD:
            mh = SRCTOP[s] == src
            if (Y[s][mh] == 1).sum() < 5 or (Y[s][mh] == 0).sum() < 5:
                print(f"  [평가불가] {src} / {s}: TP={int((Y[s][mh]==1).sum())} "
                      f"FP={int((Y[s][mh]==0).sum())}")
                continue
            p, q, cos = project(G[s][mh], vs, ts)
            ss_rows += stats_rows(s, Y[s][mh], p, q, cos, L, tag=f"same_source:{src}")
    write_csv(RES / "same_source_alignment.csv", ss_rows)
    if ss_rows:
        print("\n=== 같은 출처 내부 TP vs FP (held-out) ===")
        for r in ss_rows:
            if r["layer_to"] >= 2:
                print(f"{r['subset']:26} {r['split']:10} {r['transition']:10} "
                      f"AUROC={r['q_auroc']:.4f} [{r['q_auroc_ci_lo']:.4f},{r['q_auroc_ci_hi']:.4f}] "
                      f"n_tp={r['n_tp']} n_fp={r['n_fp']}")
    else:
        print("\n  같은 출처 방향을 만들 수 있는 출처가 없다 (TP/FP 중 한쪽이 부족).")
    print(f"\n저장 -> {RES}, {ART}")


if __name__ == "__main__":
    main()

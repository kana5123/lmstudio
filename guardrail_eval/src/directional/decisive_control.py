"""결정적 통제 (지시문 20절 Q5·Q8).

(A) cos(v_U, v_source) 를 자릿수 끝까지 본다.  1.0 이면 v_U 는 말뭉치 방향과 같다.
(B) wildchat 내부 검정력: held-out TP 가 8~11건뿐이라 부트스트랩 구간은 신뢰할 수 없다
    (9개를 복원추출해봐야 서로 다른 다중집합이 몇 개 안 나온다).
    -> **치환 검정(permutation test)** 으로 귀무분포를 직접 만든다.
    -> held-out 세 분할의 wildchat 을 합쳐 표본을 늘린다(TP 28건).
(C) 21건뿐인 학습 wildchat TP 가 방향을 만들 만한가 — 학습 TP 를 무작위로 21건 뽑은
    '가짜 방향'과 비교한다.
"""
import sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.metrics import roc_auc_score

from directional.global_direction import (load, movements, sources, fit_directions,
                                          project, write_csv, EPS)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/directional_alignment"
ART = ROOT / "artifacts/directional_alignment"
TRAIN, HELD = "ver_train", ("ver_dev", "eval_val", "eval_test")
NPERM = 20000


def main():
    SRC = sources()
    data = {s: load(s) for s in (TRAIN,) + HELD}
    G = {s: movements(d) for s, d in data.items()}
    Y = {s: d["gt"].numpy() for s, d in data.items()}
    TOP = {s: np.array([SRC[i].split("/")[0] for i in d["sample_id"]]) for s, d in data.items()}
    WC = {s: TOP[s] == "wildchat" for s in data}
    L = G[TRAIN].shape[1]
    vd = torch.load(ART / "v_u.pt", weights_only=False)
    vU, tau = vd["v"].numpy(), vd["tau"].numpy()

    # ---------- (A) ----------
    jb = ~WC[TRAIN]
    dS = G[TRAIN][jb].mean(0) - G[TRAIN][~jb].mean(0)
    vS = dS / (np.linalg.norm(dS, axis=-1, keepdims=True) + EPS)
    print("=== (A) v_U 와 '출처만으로 만든 방향' 사이의 각도 (라벨 미사용) ===")
    print(f"{'전이':10} {'cos':>18} {'각도(도)':>10}")
    rows = []
    for li in range(1, L):
        c = float(np.clip(vU[li] @ vS[li], -1, 1))
        ang = float(np.degrees(np.arccos(c)))
        rows.append({"transition": f"L{li}->L{li+1}", "cos_vU_vSource": c, "angle_deg": ang})
        print(f"L{li}->L{li+1:<6} {c:18.10f} {ang:10.4f}")
    write_csv(RES / "vU_vs_vSource_precision.csv", rows)

    # ---------- (B) wildchat 내부, held-out 합쳐 치환검정 ----------
    m = WC[TRAIN]
    vW, _, _, tauW = fit_directions(G[TRAIN][m], Y[TRAIN][m])
    Gw = np.concatenate([G[s][WC[s]] for s in HELD])
    Yw = np.concatenate([Y[s][WC[s]] for s in HELD])
    print(f"\n=== (B) wildchat 내부 (held-out 3개 분할 합침): TP={int(Yw.sum())} "
          f"FP={int((1-Yw).sum())} ===")
    print(f"  학습 방향은 ver_train wildchat (TP={int(Y[TRAIN][m].sum())} "
          f"FP={int((1-Y[TRAIN][m]).sum())}) 에서만 적합")
    _, qw, _ = project(Gw, vW, tauW)
    rng = np.random.default_rng(0)
    out = []
    n, npos = len(Yw), int(Yw.sum())
    print(f"{'전이':10} {'AUROC':>8} {'치환 p':>10} {'귀무 95%':>18}")
    for li in range(1, L):
        obs = float(roc_auc_score(Yw, qw[:, li]))
        # 빠른 치환검정: AUROC = (양성 순위합 - npos(npos+1)/2) / (npos*nneg)
        # 순위를 한 번만 구해두면 치환마다 O(npos) 로 끝난다 (roc_auc_score 재호출 불필요)
        from scipy.stats import rankdata
        rk = rankdata(qw[:, li])
        nneg = n - npos
        idx = np.argsort(rng.random((NPERM, n)), axis=1)[:, :npos]   # 치환마다 양성 위치
        null = (rk[idx].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
        p = float((null >= obs).mean())
        lo, hi = np.percentile(null, [2.5, 97.5])
        out.append({"transition": f"L{li}->L{li+1}", "n_tp": int(Yw.sum()),
                    "n_fp": int((1 - Yw).sum()), "auroc": obs, "perm_p": p,
                    "null_lo": float(lo), "null_hi": float(hi)})
        print(f"L{li}->L{li+1:<6} {obs:8.4f} {p:10.5f} [{lo:.4f},{hi:.4f}]")
    write_csv(RES / "wildchat_permutation_test.csv", out)

    # ---------- (C) 21건짜리 방향이 특별한가 ----------
    print(f"\n=== (C) 학습 wildchat TP 21건 대신 '무작위 TP 21건'으로 만든 가짜 방향 ===")
    print("    (전역 TP 풀에서 21건을 뽑아 같은 절차로 방향을 만든 뒤 wildchat held-out 평가)")
    tp_idx = np.flatnonzero(Y[TRAIN] == 1)
    fp_w = G[TRAIN][m][Y[TRAIN][m] == 0]
    fake = []
    for t in range(30):
        r2 = np.random.default_rng(100 + t)
        pick = r2.choice(tp_idx, int(Y[TRAIN][m].sum()), replace=False)
        mu_t, mu_f = G[TRAIN][pick].mean(0), fp_w.mean(0)
        d = mu_t - mu_f
        vf = d / (np.linalg.norm(d, axis=-1, keepdims=True) + EPS)
        tf = np.einsum("lh,lh->l", vf, (mu_t + mu_f) / 2)
        _, qf, _ = project(Gw, vf, tf)
        fake.append([roc_auc_score(Yw, qf[:, li]) for li in range(1, L)])
    fake = np.array(fake)
    print(f"{'전이':10} {'실제 wildchat 방향':>18} {'가짜방향 평균±표준편차':>24}")
    fr = []
    for j, li in enumerate(range(1, L)):
        obs = out[j]["auroc"]
        print(f"L{li}->L{li+1:<6} {obs:18.4f} {fake[:,j].mean():14.4f}±{fake[:,j].std():.4f}")
        fr.append({"transition": f"L{li}->L{li+1}", "auroc_wildchat_dir": obs,
                   "auroc_fake_mean": float(fake[:, j].mean()),
                   "auroc_fake_std": float(fake[:, j].std())})
    write_csv(RES / "fake_direction_control.csv", fr)
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main()

"""v_U 가 '정확성 방향'인가 '말뭉치 방향'인가 (지시문 11·20절 Q8).

이 벤치마크에서 오탐(FP)은 전부 wildchat, 정탐(TP)은 거의 전부 탈옥 모음집이다.
따라서 v_U 가 정말로 정탐/오탐을 가르는 방향인지, 아니면 그냥 두 말뭉치를 가르는
방향인지 구별해야 한다.  두 가지로 직접 잰다.

  (1) 말뭉치 방향과의 각도
      v_S^(l) = normalize( mean(g | 탈옥 말뭉치) - mean(g | wildchat) )
        <- **TP/FP 라벨을 전혀 쓰지 않고** 출처만으로 만든 방향 (ver_train 전용)
      cos( v_U^(l), v_S^(l) ) 이 1 에 가까우면 v_U 는 사실상 말뭉치 방향이다.

  (2) 같은 사영값 q 로 '출처'를 얼마나 잘 맞히나
      q 가 TP/FP 를 맞히는 AUROC 와 출처(wildchat 여부)를 맞히는 AUROC 를 비교한다.
      두 값이 같으면 q 는 정확성이 아니라 출처를 재고 있는 것이다.

  (3) wildchat 내부 전용 방향 v_U,wildchat 과 전역 v_U 의 각도
"""
import csv, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.metrics import roc_auc_score

from directional.global_direction import (load, movements, sources, fit_directions,
                                          project, write_csv, boot_auroc, EPS)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/directional_alignment"
ART = ROOT / "artifacts/directional_alignment"
TRAIN, HELD = "ver_train", ("ver_dev", "eval_val", "eval_test")


def main():
    from data.splits import sid
    SRC = sources()
    data = {s: load(s) for s in (TRAIN,) + HELD}
    G = {s: movements(d) for s, d in data.items()}
    Y = {s: d["gt"].numpy() for s, d in data.items()}
    TOP = {s: np.array([SRC[i].split("/")[0] for i in d["sample_id"]]) for s, d in data.items()}
    WC = {s: (TOP[s] == "wildchat").astype(int) for s in data}     # 1 = wildchat
    L = G[TRAIN].shape[1]

    vd = torch.load(ART / "v_u.pt", weights_only=False)
    vU, tau = vd["v"].numpy(), vd["tau"].numpy()

    # ---- (1) 출처만으로 만든 방향 (라벨 미사용) ----
    jb = WC[TRAIN] == 0
    mu_jb, mu_wc = G[TRAIN][jb].mean(0), G[TRAIN][~jb].mean(0)
    dS = mu_jb - mu_wc
    vS = dS / (np.linalg.norm(dS, axis=-1, keepdims=True) + EPS)

    # ---- (3) wildchat 내부 전용 방향 ----
    m = WC[TRAIN] == 1
    vW, _, _, tauW = fit_directions(G[TRAIN][m], Y[TRAIN][m])

    rows = []
    print("=== (1)(3) 방향 사이 각도 (코사인) ===")
    print(f"{'전이':10} {'cos(v_U, v_source)':>20} {'cos(v_U, v_U,wildchat)':>24}")
    for li in range(L):
        c_us = float(vU[li] @ vS[li])
        c_uw = float(vU[li] @ vW[li])
        rows.append({"transition": f"L{li}->L{li+1}", "layer_to": li + 1,
                     "main_analysis": bool(li + 1 >= 2),
                     "cos_vU_vSource": c_us, "cos_vU_vWildchatOnly": c_uw})
        if li >= 1:
            print(f"L{li}->L{li+1:<6} {c_us:20.4f} {c_uw:24.4f}")
    write_csv(RES / "direction_vs_source_direction.csv", rows)

    # ---- (2) q 가 TP/FP 를 맞히나, 출처를 맞히나 ----
    print("\n=== (2) 같은 q 로 'TP/FP' vs '출처(wildchat 여부)' 판별력 비교 ===")
    print(f"{'split':10} {'전이':10} {'AUROC(TP/FP)':>13} {'AUROC(출처)':>12} {'차이':>8}")
    cmp_rows = []
    for s in HELD:
        p, q, cos = project(G[s], vU, tau)
        for li in range(L):
            if li < 1:
                continue
            a_tf = float(roc_auc_score(Y[s], q[:, li]))
            # 출처: wildchat=1 이면 FP 쪽 -> 부호를 맞춰 비교 (1 - wildchat)
            a_src = float(roc_auc_score(1 - WC[s], q[:, li]))
            cmp_rows.append({"split": s, "transition": f"L{li}->L{li+1}", "layer_to": li + 1,
                             "auroc_tp_fp": a_tf, "auroc_source": a_src,
                             "diff": a_tf - a_src})
            if s == "eval_test":
                print(f"{s:10} L{li}->L{li+1:<6} {a_tf:13.4f} {a_src:12.4f} {a_tf-a_src:8.4f}")
    write_csv(RES / "tpfp_vs_source_auroc.csv", cmp_rows)

    # ---- (4) wildchat 내부 전용 방향으로 wildchat 내부 TP/FP 분리 ----
    print("\n=== (4) wildchat 전용 방향 v_U,wildchat 으로 wildchat 내부 TP/FP 분리 ===")
    ss = []
    for s in HELD:
        mh = WC[s] == 1
        ntp, nfp = int((Y[s][mh] == 1).sum()), int((Y[s][mh] == 0).sum())
        p, q, cos = project(G[s][mh], vW, tauW)
        for li in range(L):
            if li < 1:
                continue
            au = float(roc_auc_score(Y[s][mh], q[:, li]))
            lo, hi = boot_auroc(Y[s][mh], q[:, li])
            ss.append({"split": s, "subset": "wildchat_only", "transition": f"L{li}->L{li+1}",
                       "layer_to": li + 1, "n_tp": ntp, "n_fp": nfp,
                       "auroc": au, "ci_lo": lo, "ci_hi": hi})
            if s == "eval_test":
                print(f"  eval_test L{li}->L{li+1:<6} AUROC={au:.4f} [{lo:.4f},{hi:.4f}] "
                      f"n_tp={ntp} n_fp={nfp}")
    write_csv(RES / "wildchat_internal_alignment.csv", ss)
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main()

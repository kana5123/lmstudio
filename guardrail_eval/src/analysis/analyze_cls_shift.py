"""1단계 가설 검증 — TP 와 FP 사이에 층별 CLS 표현 변화 차이가 있는가.

**PCA 그림만으로 주장하지 않는다.**  모든 판단은 ver_train 에서 학습한 선형 탐침(probe)을
eval_val / eval_test 에서 평가한 수치로 내린다.

특징 후보:
    h1              1번째 층 CLS
    hL              마지막 층 CLS
    concat(h1,hL)
    delta_h         hL - h1           <- 핵심 가설
    adjacent_delta  h_l - h_(l-1) 전부 이어붙임
"""
import json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.metrics import linear_probe, auroc

ROOT = Path(__file__).resolve().parents[2]
FEAT, PLOT, RES = ROOT / "artifacts/features", ROOT / "artifacts/plots", ROOT / "artifacts/results"
SEED = 0


def load(split):
    d = torch.load(FEAT / f"hidden_{split}.pt", weights_only=False)
    return d, d["h"].numpy(), d["gt"].numpy()


def feats(h, L):
    """h: (n, L+1, H).  층 l 은 h[:, l] (0=임베딩)."""
    return {
        "h1": h[:, 1],
        "hL": h[:, L],
        "concat_h1_hL": np.concatenate([h[:, 1], h[:, L]], 1),
        "delta_h": h[:, L] - h[:, 1],
        "adjacent_delta": np.concatenate([h[:, l] - h[:, l - 1] for l in range(1, L + 1)], 1),
    }


def main():
    tr, htr, ytr = load("ver_train")
    dv, hdv, ydv = load("ver_dev")
    va, hva, yva = load("eval_val")
    te, hte, yte = load("eval_test")
    L = tr["layers"]
    print(f"층수 L={L}  ver_train {len(ytr)} (TP {ytr.sum()}/FP {(1-ytr).sum()})  "
          f"ver_dev {len(ydv)}  eval_val {len(yva)}  eval_test {len(yte)}")

    Ftr, Fdv, Fva, Fte = (feats(x, L) for x in (htr, hdv, hva, hte))
    rows = []
    for name in Ftr:
        r_te, s_te, _ = linear_probe(Ftr[name], ytr, Fte[name], yte, SEED)
        r_va, _, _ = linear_probe(Ftr[name], ytr, Fva[name], yva, SEED)
        r_dv, _, _ = linear_probe(Ftr[name], ytr, Fdv[name], ydv, SEED)
        rows.append({"feature": name, "dim": Ftr[name].shape[1],
                     "dev_auroc": r_dv["auroc"], "val_auroc": r_va["auroc"],
                     "test_auroc": r_te["auroc"], "test_auprc": r_te["auprc"],
                     "test_acc": r_te["acc"], "test_f1": r_te["f1"]})
        print(f"{name:16} dim={Ftr[name].shape[1]:5}  ver_dev AUROC={r_dv['auroc']:.4f}  "
              f"eval_val AUROC={r_va['auroc']:.4f}  eval_test AUROC={r_te['auroc']:.4f} "
              f"AUPRC={r_te['auprc']:.4f} acc={r_te['acc']:.4f} F1={r_te['f1']:.4f}")

    # --- 층별 탐침: TP/FP 정보가 어느 층부터 생기는가 ---
    per_layer = []
    for l in range(L + 1):
        r, _, _ = linear_probe(htr[:, l], ytr, hte[:, l], yte, SEED)
        rd, _, _ = linear_probe(htr[:, l], ytr, hdv[:, l], ydv, SEED)
        per_layer.append({"layer": l, "test_auroc": r["auroc"], "dev_auroc": rd["auroc"]})
        print(f"  층 {l:2} (0=임베딩)  eval_test AUROC={r['auroc']:.4f}  ver_dev AUROC={rd['auroc']:.4f}")

    # --- 기하: centroid 거리 / cos / L2 (train 에서만 계산) ---
    D = Ftr["delta_h"]
    mu_tp, mu_fp = D[ytr == 1].mean(0), D[ytr == 0].mean(0)
    v = (mu_tp - mu_fp) / np.linalg.norm(mu_tp - mu_fp)
    geo = {"centroid_l2": float(np.linalg.norm(mu_tp - mu_fp)),
           "cos_mu": float(mu_tp @ mu_fp / np.linalg.norm(mu_tp) / np.linalg.norm(mu_fp))}
    # v 로 사영한 점수의 분리력 (train 에서 만든 v 를 test 에 적용)
    Dte = Fte["delta_h"]
    geo["proj_v_test_auroc"] = auroc(yte, Dte @ v)
    geo["proj_v_dev_auroc"] = auroc(ydv, Fdv["delta_h"] @ v)
    print(f"centroid L2={geo['centroid_l2']:.3f}  cos(mu_TP,mu_FP)={geo['cos_mu']:.4f}  "
          f"v 사영 AUROC: dev={geo['proj_v_dev_auroc']:.4f} test={geo['proj_v_test_auroc']:.4f}")

    # --- 그림: PCA (해석 보조용일 뿐, 판단 근거 아님) ---
    PLOT.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (nm, Xtr_, Xte_, yte_) in zip(axes, [
            ("h_L", Ftr["hL"], Fte["hL"], yte),
            ("delta_h = h_L - h_1", Ftr["delta_h"], Fte["delta_h"], yte),
            ("h_1", Ftr["h1"], Fte["h1"], yte)]):
        sc = StandardScaler().fit(Xtr_); p = PCA(2, random_state=SEED).fit(sc.transform(Xtr_))
        Z = p.transform(sc.transform(Xte_))
        ax.scatter(Z[yte_ == 1, 0], Z[yte_ == 1, 1], s=5, alpha=.45, label="TP", c="#c0392b")
        ax.scatter(Z[yte_ == 0, 0], Z[yte_ == 0, 1], s=5, alpha=.45, label="FP", c="#2471a3")
        ax.set_title(f"{nm}  (PCA fit on ver_train, shown: eval_test)")
        ax.legend(markerscale=3)
    plt.tight_layout(); plt.savefig(PLOT / "cls_shift_pca.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["layer"] for r in per_layer], [r["test_auroc"] for r in per_layer], "o-", label="eval_test")
    ax.plot([r["layer"] for r in per_layer], [r["dev_auroc"] for r in per_layer], "s--", label="ver_dev")
    ax.axhline(.5, color="gray", ls=":"); ax.set_xlabel("layer (0=embedding)")
    ax.set_ylabel("TP/FP linear-probe AUROC"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOT / "layerwise_probe_auroc.png", dpi=130); plt.close()

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "cls_shift.json").write_text(json.dumps(
        {"features": rows, "per_layer": per_layer, "geometry": geo,
         "n": {"ver_train": len(ytr), "ver_dev": len(ydv), "eval_val": len(yva), "eval_test": len(yte)},
         "pos_rate": {"ver_train": float(ytr.mean()), "eval_test": float(yte.mean())}},
        ensure_ascii=False, indent=1))
    print(f"저장 -> {RES/'cls_shift.json'}, {PLOT}/cls_shift_pca.png, layerwise_probe_auroc.png")


if __name__ == "__main__":
    main()

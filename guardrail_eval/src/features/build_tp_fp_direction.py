"""TP/FP 깊이이동 방향 v 를 만든다.  **검증기 학습셋(ver_train)에서만** 계산한다.

핵심 특징:
    delta_h = h_L - h_1      (h_1 = 1번째 인코더 층 출력의 CLS, h_L = 마지막 층)
방향:
    v = normalize(mean(delta_h | TP) - mean(delta_h | FP))

비교용 baseline 방향도 같이 저장한다(전부 ver_train 전용):
    raw      : mu_TP - mu_FP (정규화 없음)
    centroid : 위를 L2 정규화한 것 = 주 방법
    probe    : 로지스틱 회귀 가중치 방향
    lda      : 선형판별분석(LDA) 방향
"""
import sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch

FEAT = Path(__file__).resolve().parents[2] / "artifacts" / "features"
DIRS = Path(__file__).resolve().parents[2] / "artifacts" / "directions"
SEED = 0


def delta_h(d):
    """delta_h = h_L - h_1.  h[:,0]=임베딩이므로 층 l 은 h[:,l]."""
    L = d["layers"]
    return (d["h"][:, L] - d["h"][:, 1]).numpy()


def main():
    tr = torch.load(FEAT / "hidden_ver_train.pt", weights_only=False)
    X, y = delta_h(tr), tr["gt"].numpy()
    print(f"ver_train UNSAFE {len(y)}건  TP={int(y.sum())} FP={int((1-y).sum())}  X={X.shape}")
    assert X.shape[1] == 768

    mu_tp, mu_fp = X[y == 1].mean(0), X[y == 0].mean(0)
    raw = mu_tp - mu_fp
    v = raw / np.linalg.norm(raw)
    print(f"centroid 거리 |mu_TP-mu_FP| = {np.linalg.norm(raw):.4f}")
    print(f"  |mu_TP|={np.linalg.norm(mu_tp):.4f}  |mu_FP|={np.linalg.norm(mu_fp):.4f}"
          f"  cos(mu_TP,mu_FP)={mu_tp@mu_fp/np.linalg.norm(mu_tp)/np.linalg.norm(mu_fp):.4f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)                      # 스케일러도 train 에서만 적합
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(X), y)
    w_probe = lr.coef_[0] / sc.scale_                  # 원 공간으로 되돌림
    w_probe = w_probe / np.linalg.norm(w_probe)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    w_lda = lda.coef_[0] / np.linalg.norm(lda.coef_[0])

    for n, w in (("probe", w_probe), ("lda", w_lda)):
        print(f"cos(centroid, {n}) = {float(v @ w):.4f}")

    DIRS.mkdir(parents=True, exist_ok=True)
    torch.save({"v": torch.tensor(v, dtype=torch.float32),
                "raw": torch.tensor(raw, dtype=torch.float32),
                "probe": torch.tensor(w_probe, dtype=torch.float32),
                "lda": torch.tensor(w_lda, dtype=torch.float32),
                "mu_tp": torch.tensor(mu_tp, dtype=torch.float32),
                "mu_fp": torch.tensor(mu_fp, dtype=torch.float32),
                "fit_split": "ver_train", "n_tp": int(y.sum()), "n_fp": int((1 - y).sum()),
                "feature": "delta_h = h_L - h_1"},
               DIRS / "pg2_tp_fp_delta_direction.pt")
    print(f"저장 -> {DIRS/'pg2_tp_fp_delta_direction.pt'}")


if __name__ == "__main__":
    main()

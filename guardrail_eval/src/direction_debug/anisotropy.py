"""§2 — shuffle-null 이 왜 높았는가.  '공통 평균' 설명을 검증/기각한다.

지적대로 delta = mean(A) - mean(B) 에서 **공통 평균은 원칙적으로 상쇄된다.**
따라서 높은 null cosine 의 원인은 다른 데 있어야 한다.  후보를 하나씩 잰다:

  (a) 공분산 비등방성 / 낮은 유효차원 (dominant PCs)
  (b) 치환 구성이 독립적이지 않음 (두 방향이 표본을 공유)
  (c) 클래스 크기 불균형
  (d) 구현 버그

핵심 지표:
  유효 랭크(effective rank)   r_eff = (tr Σ)^2 / tr(Σ^2)
  참여율(participation ratio) 동일 정의 (여기서는 같은 값)
  상위 k개 주성분 분산 비율
"""
import sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "artifacts/features"
RP = ROOT / "artifacts/direction_repro"
RES = ROOT / "results/direction_debug"


def movements(h):
    return h[:, 1:] - h[:, :-1]


def spectrum(G):
    """G: (n, H).  공분산 고유값 스펙트럼 지표."""
    X = G - G.mean(0, keepdims=True)
    n = len(X)
    # 고유값 = 특이값^2/(n-1)
    s = np.linalg.svd(X, compute_uv=False)
    ev = (s ** 2) / max(n - 1, 1)
    tot = ev.sum()
    r_eff = float(tot ** 2 / (ev ** 2).sum())
    return {"top1": float(ev[0] / tot), "top5": float(ev[:5].sum() / tot),
            "top10": float(ev[:10].sum() / tot), "top50": float(ev[:50].sum() / tot),
            "effective_rank": r_eff, "participation_ratio": r_eff,
            "n": int(n), "dim": int(X.shape[1])}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    import glob
    rows = []

    # --- WildJailbreak (direction_repro 에서 쓴 것과 같은 은닉표현) ---
    fs = sorted(glob.glob(str(RP / "hidden_wildjailbreak_adversarial_*of*.pt")))
    ds = [torch.load(f, weights_only=False) for f in fs]
    h = torch.cat([d["h"] for d in ds]).numpy().astype(np.float64)
    split = np.array([x for d in ds for x in d["split"]])
    G = movements(h)[split == "train"]          # TRAIN 에서만 공분산 적합
    for li in range(G.shape[1]):
        rows.append({"dataset": "wildjailbreak", "transition": f"L{li}->L{li+1}",
                     **spectrum(G[:, li])})

    # --- JailbreaksOverTime (보존된 은닉표현) ---
    d = torch.load(FEAT / "hidden_ver_train.pt", weights_only=False)
    Gj = movements(d["h"].numpy().astype(np.float64))
    for li in range(Gj.shape[1]):
        rows.append({"dataset": "jailbreaksovertime", "transition": f"L{li}->L{li+1}",
                     **spectrum(Gj[:, li])})

    df = pd.DataFrame(rows)
    df.to_csv(RES / "anisotropy_by_layer.csv", index=False)
    print("=== 이동벡터 g^(l) 의 공분산 스펙트럼 (TRAIN 전용, 차원 768) ===")
    print(f"{'dataset':20} {'전이':10} {'top1':>7} {'top5':>7} {'top10':>7} {'top50':>7} "
          f"{'유효랭크':>9}")
    for _, r in df.iterrows():
        if int(r["transition"].split("->")[0][1:]) < 1:
            continue
        print(f"{r['dataset']:20} {r['transition']:10} {r['top1']:7.3f} {r['top5']:7.3f} "
              f"{r['top10']:7.3f} {r['top50']:7.3f} {r['effective_rank']:9.2f}")
    print(f"\n저장 -> {RES/'anisotropy_by_layer.csv'}")
    print("해석: 유효랭크가 768 에 가까우면 등방, 1 에 가까우면 한 방향에 몰림.")


if __name__ == "__main__":
    main()

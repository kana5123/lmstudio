"""PHASE1-5: 증거 인터페이스 감사.

공유 검증기는 base model 마다 파라미터를 따로 둘 수 없다.  따라서 입력 특징이
"모든 base model 이 동일한 의미로 내주는 것" 이어야 한다.  무엇이 그런지 감사한다.

두 부류를 구분한다.
  (A) 원시 은닉벡터 768차원 -- 이 6개 모델이 우연히 전부 768 이라 크기는 맞지만,
      좌표축의 의미가 모델마다 다르다(기저 불일치).  같은 5번 차원이 다른 뜻이다.
  (B) 기저 없는 기하 요약 -- 노름, 코사인, 층간 이동량처럼 좌표축을 고르지 않는 값.
      차원수가 달라도 정의되고, 모델이 달라도 같은 의미를 갖는다.

(A) 가 실제로 공유 가능한지는 "모델 간 기저가 정렬돼 있는가" 로 판정한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HID = ROOT / "artifacts/shared_verifier/hidden"
RES = ROOT / "results/shared_verifier"
K = 7   # 깊이 정규화 구간 수 (0=임베딩, 1=출력)


def depth_resample(h, k=K):
    """(n, L+1, d) -> (n, k, d).  층 수가 다른 모델을 같은 깊이 눈금에 올린다.
    깊이 t = l/L 에서 선형보간.  12층/6층 모두 t=0..1 위에 놓인다."""
    n, L1, d = h.shape
    src = np.linspace(0.0, 1.0, L1)
    tgt = np.linspace(0.0, 1.0, k)
    idx = np.interp(tgt, src, np.arange(L1))
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    w = torch.from_numpy((idx - lo).astype(np.float32)).view(1, k, 1)
    return h[:, lo] * (1 - w) + h[:, hi] * w


def geom_features(h):
    """(n, L+1, d) -> 기저 없는 기하 요약 (n, F) 와 이름."""
    x = depth_resample(h.float())                       # (n, K, d)
    g = x[:, 1:] - x[:, :-1]                            # 층간 이동 (n, K-1, d)
    nrm = x.norm(dim=-1)                                # 층별 노름 (n, K)
    gn = g.norm(dim=-1)                                 # 이동 크기 (n, K-1)
    fin = x[:, -1:]                                     # 최종 표현
    cos_fin = torch.cosine_similarity(x, fin, dim=-1)   # 최종과의 정렬 (n, K)
    cos_g = torch.cosine_similarity(g[:, 1:], g[:, :-1], dim=-1)   # 이동 방향 꺾임 (n, K-2)
    feats = torch.cat([nrm, gn, cos_fin, cos_g], 1)
    names = ([f"norm_L{i}" for i in range(K)] + [f"step_L{i}" for i in range(K - 1)]
             + [f"cosfin_L{i}" for i in range(K)] + [f"bend_L{i}" for i in range(K - 2)])
    return feats.numpy(), names


def main():
    files = sorted(HID.glob("*.pt"))
    if not files:
        sys.exit("은닉표현이 아직 없다")

    # --- 인터페이스 표 -------------------------------------------------------
    rows, mean_by_model = [], {}
    for f in files:
        d = torch.load(f, weights_only=False)
        h = d["h_cls"]
        rows.append(dict(model=d["model"], dataset=d["dataset"], layers=d["layers"],
                         n_hidden_states=h.shape[1], hidden_dim=h.shape[2],
                         n_logits=d["logits"].shape[1]))
        mean_by_model.setdefault(d["model"], []).append(h.float().mean(0))
    iface = pd.DataFrame(rows).drop_duplicates(["model"]).drop(columns="dataset")
    print("=== 모델별 증거 인터페이스 ===")
    print(iface.to_string(index=False), "\n")

    # --- (A) 기저 정렬 여부: 모델 간 평균 은닉벡터 코사인 --------------------
    mk = sorted(mean_by_model)
    M = {m: depth_resample(torch.stack(mean_by_model[m]).mean(0, keepdim=True))[0] for m in mk}
    ali = []
    for i, a in enumerate(mk):
        for b in mk[i + 1:]:
            c = torch.cosine_similarity(M[a], M[b], dim=-1)      # 깊이별 코사인 (K,)
            ali.append(dict(model_a=a, model_b=b,
                            **{f"cos_d{j}": round(float(c[j]), 4) for j in range(K)}))
    ali = pd.DataFrame(ali)
    print("=== (A) 원시 768차원 기저가 모델 간 정렬돼 있는가 ===")
    print("   깊이별 평균 은닉벡터의 코사인.  0 근처면 좌표축 의미가 서로 무관하다는 뜻")
    print(ali.to_string(index=False), "\n")

    # --- (B) 기저 없는 기하 특징의 스케일 비교 -------------------------------
    gs = []
    for f in files:
        d = torch.load(f, weights_only=False)
        v, names = geom_features(d["h_cls"])
        gs.append(pd.DataFrame(v, columns=names).assign(model=d["model"], dataset=d["dataset"]))
    G = pd.concat(gs, ignore_index=True)
    print("=== (B) 기저 없는 기하 특징: 모델별 평균 ===")
    print("   모델마다 스케일이 크게 다르면 공유 검증기는 모델별 정규화가 필요해진다")
    print(G.groupby("model")[names].mean().round(3).to_string(), "\n")

    RES.mkdir(parents=True, exist_ok=True)
    iface.to_csv(RES / "evidence_interface.csv", index=False)
    ali.to_csv(RES / "evidence_basis_alignment.csv", index=False)
    G.groupby("model")[names].agg(["mean", "std"]).to_csv(RES / "evidence_geom_scale.csv")
    print(f"저장 -> {RES}/evidence_{{interface,basis_alignment,geom_scale}}.csv")


if __name__ == "__main__":
    main()


def model_identity_probe():
    """공유 검증기가 '어느 모델이 만든 벡터인가' 를 알 수 있는가.

    알 수 있다면 파라미터가 한 벌이어도 내부에서 모델별로 갈라 처리하는 것을
    학습할 수 있다 -- 즉 '본 적 있는 모델에서 잘 된다' 가 '모델 무관' 을 뜻하지 않는다.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from src.shared_verifier.features import load_all

    tab, feats, _ = load_all(0)
    tr = (tab.split == "train").to_numpy()
    y = tab.model.to_numpy()
    out = []
    for f in ("raw_last", "geom", "conf"):
        X = feats[f]
        s = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=400, n_jobs=12).fit(s.transform(X[tr][::4]), y[tr][::4])
        out.append(dict(features=f, n_models=tab.model.nunique(),
                        chance=1 / tab.model.nunique(),
                        accuracy=accuracy_score(y[~tr], c.predict(s.transform(X[~tr])))))
    df = pd.DataFrame(out)
    df.to_csv(RES / "evidence_model_identity.csv", index=False)
    print("=== 모델 정체성 복원 가능성 (높을수록 모델별 전문화가 가능) ===")
    print(df.round(4).to_string(index=False))
    return df

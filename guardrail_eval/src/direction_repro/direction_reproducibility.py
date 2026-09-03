"""TP-vs-FP 방향이 출처가 바뀌어도 재현되는가 (다음 단계 H).

검증 대상 질문:
  delta_d^(l) = mu_TP,d^(l) - mu_FP,d^(l)  가 서로 다른 d 에서 같은 방향인가?

d 를 만드는 두 경로:
  (A) CORE 출처(wildjailbreak:adversarial) **내부 분할** — 유사 cross-source.
      진짜 cross-dataset 이 아니라는 한계를 명시한다.
        random_A/B   무작위 절반  (대조군: 여기서 재현 안 되면 신호 자체가 없음 = 천장)
        topic_0..3   원문 TF-IDF k-means 군집 (내용 이동)
        len_short/long  길이 중앙값 분할 (길이 이동)
  (B) **진짜 cross-dataset** — 보존해 둔 JailbreaksOverTime 방향
      (artifacts/directional_alignment/v_u.pt, ver_train 에서만 적합) 과 비교.

모든 방향은 각 분할의 **train split 에서만** 적합하고, 전이 평가는 상대 분할의 val+test 에서 한다.

잡음 바닥 — **이론값 1/sqrt(768)≈0.036 을 쓰면 안 된다.**  g^(l) 들은 등방(isotropic)이 아니라
공통 평균 방향을 크게 공유하므로, 라벨을 무작위로 섞어 만든 두 방향도 이미 상당히 정렬된다.
따라서 귀무분포를 **실측**한다(같은 데이터에서 라벨만 섞어 절반씩 방향을 만들고 cos 측정, 20회).
실측 귀무 |cos| 평균은 층에 따라 0.30~0.68 이다.  모든 cos 값은 이 값과 비교해야 한다.
"""
import csv, glob, itertools, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/multisource_guard"
ART = ROOT / "artifacts/direction_repro"
JOT_V = ROOT / "artifacts/directional_alignment/v_u.pt"
RES = ROOT / "results/direction_repro"
EPS = 1e-12
GROUP = "wildjailbreak:adversarial"


def load_hidden():
    fs = sorted(glob.glob(str(ART / f"hidden_{GROUP.replace(':','_')}_*of*.pt")))
    assert fs, "은닉표현 없음"
    ds = [torch.load(f, weights_only=False) for f in fs]
    out = {"h": torch.cat([d["h"] for d in ds]).numpy().astype(np.float64),
           "gt": torch.cat([d["gt"] for d in ds]).numpy(),
           "text_len": torch.cat([d["text_len"] for d in ds]).numpy()}
    for k in ("sample_id", "cell", "split", "group_key"):
        out[k] = [x for d in ds for x in d[k]]
    return out


def movements(h):
    """g^(l) = h[:, l] - h[:, l-1].  색인 0 = 임베딩->L1 (주 분석 제외)."""
    return h[:, 1:] - h[:, :-1]


def fit_dir(G, y):
    tp, fp = G[y == 1], G[y == 0]
    if len(tp) < 5 or len(fp) < 5:
        return None, None
    d = tp.mean(0) - fp.mean(0)
    v = d / (np.linalg.norm(d, axis=-1, keepdims=True) + EPS)
    tau = np.einsum("lh,lh->l", v, (tp.mean(0) + fp.mean(0)) / 2)
    return v, tau


def make_partitions(D, texts):
    """의사 출처(pseudo-source) 분할.  전부 group_key 단위라 중복이 갈라지지 않는다."""
    gk = np.array(D["group_key"])
    parts = {}
    hv = np.array([int(__import__("hashlib").sha1(f"repro||{k}".encode()).hexdigest()[:8], 16)
                   for k in gk])
    parts["random_A"] = hv % 2 == 0
    parts["random_B"] = hv % 2 == 1
    L = D["text_len"]
    med = np.median(L)
    parts["len_short"] = L <= med
    parts["len_long"] = L > med
    vec = TfidfVectorizer(max_features=30000, min_df=3, stop_words="english")
    X = vec.fit_transform(texts)
    km = KMeans(4, n_init=10, random_state=0).fit(X)
    for c in range(4):
        parts[f"topic_{c}"] = km.labels_ == c
    return parts


def main():
    RES.mkdir(parents=True, exist_ok=True)
    D = load_hidden()
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[["sample_id", "text"]]
    tx = can.set_index("sample_id")["text"].to_dict()
    texts = [str(tx[s]) for s in D["sample_id"]]
    G = movements(D["h"])
    y = D["gt"]
    split = np.array(D["split"])
    nL = G.shape[1]
    print(f"{GROUP}: n={len(y)} TP={int(y.sum())} FP={int((1-y).sum())}  전이 {nL}개")

    parts = make_partitions(D, texts)
    # 각 분할의 train 에서만 방향 적합
    V, TAU, info = {}, {}, []
    for name, m in parts.items():
        tr = m & (split == "train")
        v, tau = fit_dir(G[tr], y[tr])
        n_tp, n_fp = int(y[tr].sum()), int((1 - y[tr]).sum())
        info.append({"partition": name, "n_total": int(m.sum()), "n_train": int(tr.sum()),
                     "train_TP": n_tp, "train_FP": n_fp, "fitted": v is not None})
        print(f"  {name:12} n={int(m.sum()):5} train={int(tr.sum()):5} (TP {n_tp:4}/FP {n_fp:4})"
              f"  {'적합' if v is not None else '표본부족'}")
        if v is not None:
            V[name], TAU[name] = v, tau
    pd.DataFrame(info).to_csv(RES / "partition_summary.csv", index=False)

    # ---------- 잡음 바닥: 라벨을 섞어 만든 방향과의 cos ----------
    rng = np.random.default_rng(0)
    tr_all = split == "train"
    null_cos = []
    for _ in range(20):
        ys = rng.permutation(y[tr_all])
        v1, _ = fit_dir(G[tr_all][:len(ys) // 2], ys[:len(ys) // 2])
        v2, _ = fit_dir(G[tr_all][len(ys) // 2:], ys[len(ys) // 2:])
        if v1 is not None and v2 is not None:
            null_cos.append(np.einsum("lh,lh->l", v1, v2))
    null_cos = np.array(null_cos)
    print("\n=== 실측 귀무분포 (라벨 무작위 섞기 20회, 같은 데이터 절반씩) ===")
    print("  " + " ".join(f"L{li}->{li+1}:{np.abs(null_cos[:,li]).mean():.3f}" for li in range(1, nL)))
    print("  -> 이론값 0.036 이 아니라 이 값과 비교해야 한다 (g 가 등방이 아님)")
    pd.DataFrame([{"transition": f"L{li}->L{li+1}",
                   "null_abs_cos_mean": float(np.abs(null_cos[:, li]).mean()),
                   "null_abs_cos_p95": float(np.percentile(np.abs(null_cos[:, li]), 95))}
                  for li in range(nL)]).to_csv(RES / "null_cosine_distribution.csv", index=False)

    # ---------- (A) 분할 간 방향 일치 ----------
    rows = []
    print(f"\n=== (A) 내부 분할 간 방향 일치 cos(v_a, v_b)  [주 전이 L1->L2 … L11->L12] ===")
    print(f"{'분할 쌍':28} " + " ".join(f"L{li}" for li in range(1, nL)))
    for a, b in itertools.combinations(sorted(V), 2):
        cs = np.einsum("lh,lh->l", V[a], V[b])
        rows.append({"pair": f"{a}|{b}", **{f"cos_L{li}->L{li+1}": float(cs[li])
                                            for li in range(nL)}})
        print(f"{a+' | '+b:28} " + " ".join(f"{cs[li]:+.2f}" for li in range(1, nL)))
    pd.DataFrame(rows).to_csv(RES / "internal_partition_cosine.csv", index=False)

    # ---------- 전이 평가: a 에서 적합한 방향을 b 의 held-out 에 적용 ----------
    tr_rows = []
    print(f"\n=== 전이 AUROC: 분할 a 의 방향을 분할 b 의 val+test 에 적용 ===")
    print(f"{'a -> b':28} " + " ".join(f"L{li}" for li in range(1, nL)))
    for a in sorted(V):
        for b in sorted(V):
            m = parts[b] & (split != "train")
            if m.sum() < 50 or y[m].sum() < 10 or (1 - y[m]).sum() < 10:
                continue
            q = np.einsum("lh,nlh->nl", V[a], G[m]) - TAU[a][None, :]
            au = [float(roc_auc_score(y[m], q[:, li])) for li in range(nL)]
            tr_rows.append({"fit_on": a, "eval_on": b, "n_eval": int(m.sum()),
                            **{f"auroc_L{li}->L{li+1}": au[li] for li in range(nL)}})
            if a != b:
                print(f"{a+' -> '+b:28} " + " ".join(f"{au[li]:.3f}" for li in range(1, nL)))
    pd.DataFrame(tr_rows).to_csv(RES / "internal_transfer_auroc.csv", index=False)

    # ---------- (B) 진짜 cross-dataset: JailbreaksOverTime 방향과 비교 ----------
    jot = torch.load(JOT_V, weights_only=False)
    vJ = jot["v"].numpy()
    assert vJ.shape[0] == nL, (vJ.shape, nL)
    vW, tauW = fit_dir(G[tr_all], y[tr_all])
    cs = np.einsum("lh,lh->l", vW, vJ)
    print(f"\n=== (B) ★ 진짜 cross-dataset: cos(v_wildjailbreak, v_JailbreaksOverTime) ===")
    print(f"{'전이':10} {'cos':>8} {'각도(도)':>9} {'실측귀무|cos|':>12} {'귀무초과':>8}")
    xr = []
    for li in range(nL):
        c = float(np.clip(cs[li], -1, 1))
        xr.append({"transition": f"L{li}->L{li+1}", "cos": c,
                   "angle_deg": float(np.degrees(np.arccos(c))),
                   "null_abs_cos_mean": float(np.abs(null_cos[:, li]).mean())})
        if li >= 1:
            nu = float(np.abs(null_cos[:, li]).mean())
            print(f"L{li}->L{li+1:<5} {c:+8.4f} {np.degrees(np.arccos(c)):9.2f} {nu:12.3f} "
                  f"{'예' if c > nu else '아니오':>8}")
    pd.DataFrame(xr).to_csv(RES / "cross_dataset_cosine.csv", index=False)

    # JOT 방향을 WildJailbreak held-out 에 적용
    ho = split != "train"
    tJ = np.einsum("lh,lh->l", vJ, (G[tr_all][y[tr_all] == 1].mean(0)
                                    + G[tr_all][y[tr_all] == 0].mean(0)) / 2)
    qJ = np.einsum("lh,nlh->nl", vJ, G[ho]) - tJ[None, :]
    qW = np.einsum("lh,nlh->nl", vW, G[ho]) - tauW[None, :]
    print(f"\n=== JOT 방향 vs 자체 방향을 WildJailbreak held-out({int(ho.sum())}건)에 적용 ===")
    print(f"{'전이':10} {'JOT 방향':>9} {'자체 방향':>10}")
    tr2 = []
    for li in range(nL):
        aJ = float(roc_auc_score(y[ho], qJ[:, li]))
        aW = float(roc_auc_score(y[ho], qW[:, li]))
        tr2.append({"transition": f"L{li}->L{li+1}", "auroc_jot_direction": aJ,
                    "auroc_own_direction": aW})
        if li >= 1:
            print(f"L{li}->L{li+1:<5} {aJ:9.4f} {aW:10.4f}")
    pd.DataFrame(tr2).to_csv(RES / "cross_dataset_transfer_auroc.csv", index=False)
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main()

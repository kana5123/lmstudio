"""§1 — direction_repro 파이프라인 감사.  보고된 수치를 저장 artifact 에서 재계산한다."""
import glob, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
RP_A = ROOT / "artifacts/direction_repro"
RP_R = ROOT / "results/direction_repro"
DA = ROOT / "artifacts/directional_alignment"
MS = ROOT / "data/multisource_guard"
FEAT = ROOT / "artifacts/features"
EPS = 1e-12
CHK = []


def ck(name, ok, detail):
    CHK.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'OK ' if ok else '주의'}] {name}: {detail}")


def main():
    fs = sorted(glob.glob(str(RP_A / "hidden_wildjailbreak_adversarial_*of*.pt")))
    ds = [torch.load(f, weights_only=False) for f in fs]
    h = torch.cat([d["h"] for d in ds]).numpy().astype(np.float64)
    gt = torch.cat([d["gt"] for d in ds]).numpy()
    sid = np.array([x for d in ds for x in d["sample_id"]])
    cell = np.array([x for d in ds for x in d["cell"]])
    split = np.array([x for d in ds for x in d["split"]])
    gk = np.array([x for d in ds for x in d["group_key"]])
    G = h[:, 1:] - h[:, :-1]

    print("=== 1. TP/FP 가 GT + PG2 hard prediction 으로 만들어졌는가 ===")
    cc = pd.read_parquet(MS / "confusion_cells.parquet").set_index("sample_id")
    sub = cc.loc[sid]
    recomputed = np.where((sub["binary_main_label"] == "UNSAFE") & (sub["pg2_prediction"] == "UNSAFE"), "TP",
                 np.where((sub["binary_main_label"] == "SAFE") & (sub["pg2_prediction"] == "UNSAFE"), "FP",
                 np.where((sub["binary_main_label"] == "SAFE") & (sub["pg2_prediction"] == "SAFE"), "TN", "FN")))
    ck("cell 재계산 일치", bool((recomputed == cell).all()),
       f"{int((recomputed==cell).sum())}/{len(cell)} 일치")
    ck("gt 필드 == (binary_main_label==UNSAFE)",
       bool((gt == (sub['binary_main_label'] == 'UNSAFE').values.astype(int)).all()),
       "GT 는 정답 라벨이지 예측이 아님")

    print("\n=== 2. 방향 적합에 held-out 이 섞였는가 ===")
    tr = split == "train"
    mu_tp = G[tr & (gt == 1)].mean(0); mu_fp = G[tr & (gt == 0)].mean(0)
    v_tr = (mu_tp - mu_fp) / (np.linalg.norm(mu_tp - mu_fp, axis=-1, keepdims=True) + EPS)
    mu_tp_all = G[gt == 1].mean(0); mu_fp_all = G[gt == 0].mean(0)
    v_all = (mu_tp_all - mu_fp_all) / (np.linalg.norm(mu_tp_all - mu_fp_all, axis=-1, keepdims=True) + EPS)
    same = float(np.einsum("lh,lh->l", v_tr, v_all).min())
    xd = pd.read_csv(RP_R / "cross_dataset_cosine.csv")
    jot = torch.load(DA / "v_u.pt", weights_only=False)
    vJ = jot["v"].numpy()
    cs_tr = np.einsum("lh,lh->l", v_tr, vJ)
    d = float(np.abs(cs_tr - xd["cos"].values).max())
    ck("보고된 cross-dataset cos 가 TRAIN 전용 방향으로 재계산됨", d < 1e-9,
       f"최대차 {d:.2e} (전체 데이터로 적합했다면 불일치했을 것; cos(v_train,v_all) 최소 {same:.4f})")

    print("\n=== 3. 분할 정의에 test 분포 정보가 쓰였는가 ===")
    ck("주제 군집(TF-IDF+KMeans) 적합 범위", False,
       "★ direction_repro.make_partitions 는 held-out 포함 **전체** 텍스트로 "
       "TfidfVectorizer/KMeans 를 적합한다. 라벨은 안 쓰므로 AUROC 를 직접 부풀리진 않으나 "
       "엄격한 규율 위반이며, 분할 정의가 test 분포를 본다")
    ck("길이 분할 기준(중앙값) 적합 범위", False,
       "★ 동일 문제 — 전체 데이터 중앙값을 씀. train 중앙값으로 해야 함")

    print("\n=== 4. duplicate / paired 가 split 을 넘나드는가 ===")
    dup = pd.DataFrame({"gk": gk, "split": split}).groupby("gk")["split"].nunique()
    ck("group_key 가 split 을 넘나들지 않음", bool((dup <= 1).all()),
       f"위반 {int((dup>1).sum())}개")

    print("\n=== 5. WildJailbreak split 별 cell 수 ===")
    t = pd.crosstab(split, cell)
    print(t.to_string())
    ck("held-out 에 TP/FP 충분", bool(t.loc[["val", "test"]].sum().min() >= 100),
       t.loc[["val", "test"]].sum().to_dict())

    print("\n=== 6. JOT v_u.pt 는 어떤 표본으로 적합됐는가 ===")
    ck("v_u.pt fit_split", jot["fit_split"] == "ver_train",
       f"fit_split={jot['fit_split']}, n_TP={jot['n_tp']}, n_FP={jot['n_fp']}, "
       f"kind={jot.get('kind')}")
    hj = torch.load(FEAT / "hidden_ver_train.pt", weights_only=False)
    Gj = hj["h"].numpy().astype(np.float64); Gj = Gj[:, 1:] - Gj[:, :-1]
    yj = hj["gt"].numpy()
    mtp, mfp = Gj[yj == 1].mean(0), Gj[yj == 0].mean(0)
    vj2 = (mtp - mfp) / (np.linalg.norm(mtp - mfp, axis=-1, keepdims=True) + EPS)
    ck("v_u.pt 재계산 일치", float(np.abs(np.einsum("lh,lh->l", vj2, vJ) - 1).max()) < 1e-9,
       f"cos 최소 {float(np.einsum('lh,lh->l', vj2, vJ).min()):.10f}")

    print("\n=== 7. 모든 cos 에서 방향 부호가 TP-FP 로 일관되는가 ===")
    ck("WildJailbreak 방향 부호", True, "mu_TP - mu_FP (코드 fit_dir: tp.mean(0)-fp.mean(0))")
    ck("JOT 방향 부호", True, "mu_TP - mu_FP (global_direction.fit_directions 동일)")
    ck("두 방향 모두 CORRECT-INCORRECT 인가", False,
       "★ UNSAFE 가지에서는 TP=correct, FP=incorrect 이므로 CORRECT-INCORRECT 가 맞다. "
       "그러나 SAFE 가지(TN/FN)는 이번 단계 전까지 아예 계산하지 않았다")

    pd.DataFrame(CHK).to_csv(ROOT / "results/direction_debug/pipeline_audit.csv", index=False)
    n_fail = sum(1 for c in CHK if not c["pass"])
    print(f"\n총 {len(CHK)}개 점검 중 주의 {n_fail}개 -> results/direction_debug/pipeline_audit.csv")


if __name__ == "__main__":
    main()

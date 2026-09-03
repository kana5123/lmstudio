"""혼동 셀 생성 + 방향학습 적합성 + 부분출처 교란 감사 (지시문 10~14·18절).

혼동 셀은 **코드에서 ground truth 와 prediction 을 직접 비교**해 만든다.
데이터셋 이름을 보고 TP/FP 를 추정하지 않는다.
"""
import glob, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, ART, RES = (ROOT / "data/multisource_guard", ROOT / "artifacts/multisource_guard",
                  ROOT / "results/multisource_guard")

# 지시문 12절: 아래 숫자는 **이론적 임계값이 아니라 실무 heuristic** 이다.
STRONG_N, USABLE_N = 50, 20


def cells(df):
    gt = df["binary_main_label"].values
    pr = df["pg2_prediction"].values
    c = np.where((gt == "UNSAFE") & (pr == "UNSAFE"), "TP",
        np.where((gt == "SAFE") & (pr == "UNSAFE"), "FP",
        np.where((gt == "SAFE") & (pr == "SAFE"), "TN", "FN")))
    return c


def table(df, key):
    rows = []
    for k, g in df.groupby(key):
        n = g["confusion_cell"].value_counts()
        TP, FP, TN, FN = (int(n.get(x, 0)) for x in ("TP", "FP", "TN", "FN"))
        att, ben = TP + FN, FP + TN
        uns = TP + FP
        rows.append({key: k, "gt_attack": att, "gt_benign": ben,
                     "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                     "TPR": TP / att if att else np.nan,
                     "FPR": FP / ben if ben else np.nan,
                     "precision": TP / uns if uns else np.nan,
                     "fp_rate_among_pg2_unsafe": FP / uns if uns else np.nan,
                     "tp_fp_ratio": (TP / FP) if FP else np.inf if TP else np.nan,
                     "total_pg2_unsafe": uns, "n": len(g)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    fs = sorted(glob.glob(str(ART / "pg2_pred_*of*.parquet")))
    assert fs, "PG2 예측 결과 없음"
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["confusion_cell"] = cells(df)
    print(f"총 {len(df)}건  " + df["confusion_cell"].value_counts().to_dict().__str__())
    df.to_parquet(DATA / "confusion_cells.parquet", index=False)

    t_src = table(df, "source_group"); t_src.to_csv(RES / "confusion_by_source.csv", index=False)
    table(df, "canonical_dataset").to_csv(RES / "confusion_by_dataset.csv", index=False)
    table(df, "canonical_label").to_csv(RES / "confusion_by_category.csv", index=False)

    print("\n=== source_group 별 혼동 셀 ===")
    print(f"{'source_group':44} {'GT공격':>7} {'GT정상':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} "
          f"{'TPR':>6} {'FPR':>6}")
    for _, r in t_src.iterrows():
        print(f"{r['source_group']:44} {r['gt_attack']:7d} {r['gt_benign']:7d} "
              f"{r['TP']:6d} {r['FP']:6d} {r['TN']:6d} {r['FN']:6d} "
              f"{r['TPR']:6.3f} {r['FPR']:6.3f}")

    # ---------- 방향학습 적합성 (지시문 12·13절) ----------
    rows = []
    for _, r in t_src.iterrows():
        TP, FP = int(r["TP"]), int(r["FP"])
        if TP == 0 or FP == 0:
            verdict = ("UNSUITABLE_FOR_DIRECTION" if (r["gt_attack"] == 0 or r["gt_benign"] == 0)
                       else "UNSUITABLE_FOR_DIRECTION")
            why = ("이 출처에는 한쪽 라벨만 있다" if (r["gt_attack"] == 0 or r["gt_benign"] == 0)
                   else f"양쪽 라벨은 있으나 PG2 가 {'FP' if FP==0 else 'TP'} 를 만들지 않음")
        elif TP >= STRONG_N and FP >= STRONG_N:
            verdict, why = "GOOD", f"TP {TP} / FP {FP} 둘 다 {STRONG_N} 이상"
        elif TP >= USABLE_N and FP >= USABLE_N:
            verdict, why = "MARGINAL", f"TP {TP} / FP {FP} — 진단용으로만"
        else:
            verdict, why = "UNSUITABLE_FOR_DIRECTION", f"TP {TP} / FP {FP} — centroid 불안정"
        rows.append({"source_group": r["source_group"], "N_TP": TP, "N_FP": FP,
                     "N_TN": int(r["TN"]), "N_FN": int(r["FN"]),
                     "class_imbalance_tp_over_fp": (TP / FP) if FP else np.nan,
                     "bootstrap_feasible": bool(TP >= USABLE_N and FP >= USABLE_N),
                     "same_source_tp_fp_possible": bool(TP > 0 and FP > 0),
                     "suitability": verdict, "reason": why})
    su = pd.DataFrame(rows).sort_values(["suitability", "N_TP"], ascending=[True, False])
    su.to_csv(RES / "source_suitability.csv", index=False)
    print("\n=== 방향학습 적합성 (임계 50/20 은 이론값이 아니라 실무 heuristic) ===")
    for _, r in su.iterrows():
        if r["suitability"] != "UNSUITABLE_FOR_DIRECTION":
            print(f"  {r['suitability']:10} {r['source_group']:44} TP={r['N_TP']:5} FP={r['N_FP']:5}"
                  f"  — {r['reason']}")

    # ---------- 부분출처 교란 감사 (지시문 14절) ----------
    can = pd.read_parquet(DATA / "canonical_samples.parquet")[
        ["sample_id", "original_source", "canonical_label", "attack_family"]]
    d2 = df.drop(columns=[c for c in ("original_source", "canonical_label", "attack_family")
                          if c in df.columns]).merge(can, on="sample_id", how="left")
    aud = []
    print("\n=== 부분출처 교란 감사: 같은 source_group 안에서 TP 와 FP 가 다른 하위출처인가 ===")
    for gname in su[su["same_source_tp_fp_possible"]]["source_group"]:
        g = d2[d2["source_group"] == gname]
        # canonical_label 은 binary_main_label 을 결정하므로 TVD 가 항상 1.0 이 나온다.
        # 그건 교란의 증거가 아니라 정의상 자명한 값이라 **감사에서 뺀다**.
        # 라벨에서 파생되지 않은 필드만 본다.
        # canonical_label 과 attack_family 는 binary_main_label 에서 파생되므로 TVD 가
        # 항상 1.0 이다 — 교란의 증거가 아니라 정의상 자명한 값이라 **감사에서 뺀다**.
        # 라벨에서 파생되지 않은 필드만 본다.
        for col in ("original_source", "language"):
            ct = pd.crosstab(g[col], g["confusion_cell"])
            for c in ("TP", "FP"):
                if c not in ct.columns:
                    ct[c] = 0
            tot_tp, tot_fp = ct["TP"].sum(), ct["FP"].sum()
            if tot_tp == 0 or tot_fp == 0:
                continue
            # 교란 지표: TP 분포와 FP 분포의 총변동거리 (1 이면 완전히 다른 하위출처)
            ptp, pfp = ct["TP"] / tot_tp, ct["FP"] / tot_fp
            tvd = float(0.5 * (ptp - pfp).abs().sum())
            aud.append({"source_group": gname, "subgroup_field": col,
                        "n_subgroups": int(ct.shape[0]), "n_TP": int(tot_tp), "n_FP": int(tot_fp),
                        "tv_distance_TP_vs_FP": tvd,
                        "confound_risk": "HIGH" if tvd > 0.9 else "MEDIUM" if tvd > 0.5 else "LOW",
                        "detail": json.dumps(
                            {str(k): {"TP": int(ct.loc[k, "TP"]), "FP": int(ct.loc[k, "FP"])}
                             for k in ct.index}, ensure_ascii=False)})
            print(f"  {gname:44} 하위={col:16}({ct.shape[0]:2}개) TVD={tvd:.3f} "
                  f"{aud[-1]['confound_risk']}")
    pd.DataFrame(aud).to_csv(RES / "source_subgroup_audit.csv", index=False)
    print(f"\n저장 -> {RES}")


if __name__ == "__main__":
    main()
